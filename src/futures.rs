//! The futures of the asynchronous operations (`WGPUFuture`), `wgpuInstanceWaitAny` and the
//! callback modes of `webgpu.h` (branch `progpu/wgpu-30` of wieslawsoltes/wgpu-native).
//!
//! Every future has an identifier of its own, issued in increasing order from one (zero is the null
//! future). An operation registers its future before it starts, and wgpu-core's completion of it
//! (a closure called while a device is polled or a queue submits) marks it completed and keeps its
//! callback until the callback mode allows it to run:
//!
//! - `AllowSpontaneous` (and an unset mode, as before futures were tracked): at once, on the thread
//!   that completed it;
//! - `AllowProcessEvents`: in `wgpuInstanceProcessEvents` or a `wgpuInstanceWaitAny` that waits
//!   for it;
//! - `WaitAnyOnly`: in a `wgpuInstanceWaitAny` that waits for it.
//!
//! A future a wait has observed is retired: its record is removed and a later wait for its
//! identifier reports it completed (a host may poll one submission again, as the ProGPU engine
//! does). The records of completed futures nobody waits for stay bounded: past
//! [`UNOBSERVED_LIMIT`] the oldest are retired too, and their callbacks run in the next
//! `wgpuInstanceProcessEvents` or `wgpuInstanceWaitAny` (of any instance).
//!
//! A timed wait (a timeout other than zero) waits for at most [`TIMED_WAIT_ANY_MAX_COUNT`]
//! futures: an infinite timeout waits for the devices of the instance to finish their work, a
//! finite one polls them until the deadline.

use crate::native;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The most futures one timed `wgpuInstanceWaitAny` waits for (`timedWaitAnyMaxCount`).
pub const TIMED_WAIT_ANY_MAX_COUNT: usize = 64;

/// The most completed futures kept for a wait that has not come; the oldest are retired beyond.
pub const UNOBSERVED_LIMIT: usize = 1024;

type Callback = Box<dyn FnOnce() + Send>;

struct Entry {
    mode: native::WGPUCallbackMode,
    completed: bool,
    callback: Option<Callback>,
}

struct Registry {
    entries: BTreeMap<u64, Entry>,
    unobserved: usize,
    retired_callbacks: Vec<Callback>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    entries: BTreeMap::new(),
    unobserved: 0,
    retired_callbacks: Vec::new(),
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn spontaneous(mode: native::WGPUCallbackMode) -> bool {
    mode != native::WGPUCallbackMode_WaitAnyOnly && mode != native::WGPUCallbackMode_AllowProcessEvents
}

/// Registers the future of an operation that is about to start, with the mode of its callback.
pub fn begin(mode: native::WGPUCallbackMode) -> native::WGPUFuture {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    registry().entries.insert(id, Entry { mode, completed: false, callback: None });
    native::WGPUFuture { id }
}

/// The future of an operation that completed (and called its callback) before returning: a wait
/// for it reports it completed.
pub fn completed() -> native::WGPUFuture {
    native::WGPUFuture { id: NEXT_ID.fetch_add(1, Ordering::Relaxed) }
}

/// Completes `future` with its callback: called now or kept for the wait its mode requires.
pub fn complete(future: native::WGPUFuture, callback: Callback) {
    let mut registry = registry();
    let Some(entry) = registry.entries.get_mut(&future.id) else {
        drop(registry);
        callback();
        return;
    };
    entry.completed = true;
    if spontaneous(entry.mode) {
        registry.entries.remove(&future.id);
        drop(registry);
        callback();
        return;
    }
    entry.callback = Some(callback);
    registry.unobserved += 1;
    if registry.unobserved > UNOBSERVED_LIMIT {
        retire_oldest(&mut registry);
    }
}

/// Retires the older half of the completed futures nobody has waited for.
fn retire_oldest(registry: &mut Registry) {
    let count = registry.unobserved / 2;
    let ids: Vec<u64> =
        registry.entries.iter().filter(|(_, entry)| entry.completed).map(|(&id, _)| id).take(count).collect();
    for id in ids {
        if let Some(callback) = registry.entries.remove(&id).and_then(|entry| entry.callback) {
            registry.retired_callbacks.push(callback);
        }
        registry.unobserved -= 1;
    }
}

/// Runs the callbacks of retired futures and, for `wgpuInstanceProcessEvents`, of the completed
/// futures whose mode allows it.
pub fn process_events() {
    let callbacks = {
        let mut registry = registry();
        let mut callbacks = std::mem::take(&mut registry.retired_callbacks);
        let ids: Vec<u64> = registry
            .entries
            .iter()
            .filter(|(_, entry)| entry.completed && entry.mode == native::WGPUCallbackMode_AllowProcessEvents)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            if let Some(callback) = registry.entries.remove(&id).and_then(|entry| entry.callback) {
                callbacks.push(callback);
            }
            registry.unobserved -= 1;
        }
        callbacks
    };
    for callback in callbacks {
        callback();
    }
}

/// Marks the completed futures of `waits` and runs their callbacks (then retires them). `None`
/// when one of them is not a future of this library; otherwise whether one is completed.
fn observe(waits: &mut [native::WGPUFutureWaitInfo]) -> Option<bool> {
    let issued = NEXT_ID.load(Ordering::Relaxed);
    let mut any = false;
    let callbacks = {
        let mut registry = registry();
        let mut callbacks = std::mem::take(&mut registry.retired_callbacks);
        for wait in waits.iter_mut() {
            let id = wait.future.id;
            if id == 0 || id >= issued {
                return None;
            }
            let completed = match registry.entries.get(&id) {
                None => true,
                Some(entry) => entry.completed,
            };
            if completed {
                if let Some(entry) = registry.entries.remove(&id) {
                    if let Some(callback) = entry.callback {
                        callbacks.push(callback);
                    }
                    registry.unobserved -= 1;
                }
                wait.completed = 1;
                any = true;
            } else {
                wait.completed = 0;
            }
        }
        callbacks
    };
    for callback in callbacks {
        callback();
    }
    Some(any)
}

/// `wgpuInstanceWaitAny` over the devices of `context`.
pub fn wait_any(context: &crate::Context, waits: &mut [native::WGPUFutureWaitInfo], timeout_ns: u64) -> native::WGPUWaitStatus {
    if waits.is_empty() {
        return native::WGPUWaitStatus_Success;
    }
    if timeout_ns != 0 && waits.len() > TIMED_WAIT_ANY_MAX_COUNT {
        return native::WGPUWaitStatus_Error;
    }
    let deadline = (timeout_ns != u64::MAX).then(|| Instant::now() + Duration::from_nanos(timeout_ns));
    match observe(waits) {
        None => return native::WGPUWaitStatus_Error,
        Some(true) => return native::WGPUWaitStatus_Success,
        Some(false) => {}
    }
    // An infinite wait waits for the queues; the rounds bound a future no queue will complete.
    let mut idle_rounds = 0;
    loop {
        let force_wait = deadline.is_none();
        let queues_empty = match context.poll_all_devices(force_wait) {
            Ok(queues_empty) => queues_empty,
            Err(_) => return native::WGPUWaitStatus_Error,
        };
        match observe(waits) {
            None => return native::WGPUWaitStatus_Error,
            Some(true) => return native::WGPUWaitStatus_Success,
            Some(false) => {}
        }
        match deadline {
            Some(deadline) => {
                if Instant::now() >= deadline {
                    return native::WGPUWaitStatus_TimedOut;
                }
                std::thread::yield_now();
            }
            None => {
                if queues_empty {
                    idle_rounds += 1;
                    if idle_rounds > 2 {
                        return native::WGPUWaitStatus_Error;
                    }
                }
            }
        }
    }
}
