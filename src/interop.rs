//! Native-object interop for hosts that own the native GPU objects (the ProGPU engine in FerroUI,
//! branch `progpu/wgpu-30` of wieslawsoltes/wgpu-native).
//!
//! A window platform that presents its own frames (a `CAMetalLayer` whose drawables it presents
//! inside the window's Core Animation transaction, so a resizing window never shows a stretched
//! frame) hands its renderer the Metal device and each frame's texture. These functions make a
//! wgpu device on that Metal device, with a command queue the host gives (one of that device),
//! and a wgpu texture of such a texture, through wgpu-hal, as the `wgpu` crate's
//! `Adapter::create_device_from_hal` and `Device::create_texture_from_hal` do.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::conv::{map_device_descriptor, map_texture_format};
use crate::native;
use crate::utils::get_base_device_limits_from_adapter_limits;
use crate::{
    DeviceLostCallback, ErrorSinkRaw, QueueId, TextureData, WGPUAdapterImpl, WGPUDeviceImpl, WGPUTextureImpl,
    DEFAULT_DEVICE_LOST_HANDLER,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLTexture, MTLTextureType};
use parking_lot::Mutex;
use std::sync::{atomic, Arc};

/// The adapter of `context` whose Metal device is `device` (the same device of the system: the same
/// registry identifier), or the first Metal adapter.
fn metal_adapter(context: &crate::Context, device: &ProtocolObject<dyn MTLDevice>) -> Option<wgc::id::AdapterId> {
    let registry_id = device.registryID();
    let adapters = context.enumerate_adapters(wgt::Backends::METAL, false);
    let mut first = None;
    for adapter in adapters {
        // SAFETY: the adapter is a Metal adapter (enumerated for the Metal backend).
        let matches = unsafe { context.adapter_as_hal::<hal::api::Metal>(adapter) }
            .is_some_and(|hal_adapter| hal_adapter.raw_device().registryID() == registry_id);
        if matches {
            return Some(adapter);
        }
        first.get_or_insert(adapter);
    }
    first
}

/// Creates a device of the Metal adapter of `instance` for the host's `id<MTLDevice>`, with the
/// host's `id<MTLCommandQueue>` of that device as its queue (both retained by the device until it
/// is released). The descriptor is that of `wgpuAdapterRequestDevice` (required features and
/// limits, the device-lost and uncaptured-error callbacks; null for the defaults). Returns null
/// when the instance has no Metal adapter or the device cannot be created (the reason is printed
/// to the standard error).
///
/// The host chooses the queue: a queue of its own for this device (with room for as many command
/// buffers as wgpu leaves uncommitted until a submission), ordered against the host's presenting
/// queue with a shared event, as FerroUI's Vello backend does.
///
/// # Safety
/// `mtl_device` and `mtl_queue` are a live `id<MTLDevice>` and an `id<MTLCommandQueue>` of it.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceCreateDeviceFromMetal(
    instance: native::WGPUInstance,
    mtl_device: *mut std::ffi::c_void,
    mtl_queue: *mut std::ffi::c_void,
    descriptor: Option<&native::WGPUDeviceDescriptor>,
) -> native::WGPUDevice {
    let instance = instance.as_ref().expect("invalid instance");
    let context = &instance.context;
    let (Some(raw_device), Some(raw_queue)) = (
        Retained::retain(mtl_device.cast::<ProtocolObject<dyn MTLDevice>>()),
        Retained::retain(mtl_queue.cast::<ProtocolObject<dyn MTLCommandQueue>>()),
    ) else {
        eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: a null Metal device or queue");
        return std::ptr::null_mut();
    };
    let Some(adapter_id) = metal_adapter(context, &raw_device) else {
        eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: the instance has no Metal adapter");
        return std::ptr::null_mut();
    };
    // Released after the device is created (the device keeps what it needs of the adapter).
    let adapter = Arc::new(WGPUAdapterImpl { context: context.clone(), id: adapter_id });
    let base_limits = get_base_device_limits_from_adapter_limits(&context.adapter_limits(adapter_id));
    let (desc, error_callback, device_lost_handler) = match descriptor {
        Some(descriptor) => {
            let (desc, error_callback) = crate::follow_chain!(
                map_device_descriptor((descriptor, base_limits),
                WGPUSType_DeviceExtras => native::WGPUDeviceExtras)
            );
            let handler = DeviceLostCallback {
                callback: descriptor.deviceLostCallbackInfo.callback,
                userdata: crate::new_userdata!(descriptor.deviceLostCallbackInfo),
            };
            (desc, error_callback, handler)
        }
        None => (
            wgt::DeviceDescriptor { required_limits: base_limits, ..Default::default() },
            None,
            DEFAULT_DEVICE_LOST_HANDLER,
        ),
    };
    let open = hal::OpenDevice::<hal::api::Metal> {
        device: hal::metal::Device::device_from_raw(raw_device, desc.required_features, &desc.required_limits),
        queue: hal::metal::Queue::queue_from_raw(raw_queue, 1.0),
    };
    let result = context.create_device_from_hal(adapter_id, open.into(), &desc, None, None);
    drop(adapter);
    match result {
        Ok((device_id, queue_id)) => Arc::into_raw(Arc::new_cyclic(|device| {
            let mut error_sink = ErrorSinkRaw::new(device_lost_handler, device.clone());
            if let Some(error_callback) = error_callback {
                error_sink.uncaptured_handler = error_callback;
            }
            WGPUDeviceImpl {
                context: context.clone(),
                id: device_id,
                queue: Arc::new(QueueId { context: context.clone(), id: queue_id }),
                error_sink: Arc::new(Mutex::new(error_sink)),
            }
        })),
        Err(error) => {
            eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: {error}");
            std::ptr::null_mut()
        }
    }
}

/// Creates a texture of the host's two-dimensional `id<MTLTexture>` (retained by the texture until
/// it is released) on a device made by [`wgpuInstanceCreateDeviceFromMetal`] for the texture's
/// device. The descriptor states what the Metal texture is: its size (depth one), format and usage,
/// one mip level and one sample. Its content is treated as uninitialized (a frame replaces it).
/// Returns null when the descriptor does not describe a texture wgpu can wrap.
///
/// # Safety
/// `mtl_texture` is a live `id<MTLTexture>` of the device's Metal device whose properties are the
/// descriptor's; its owner does not destroy it while the texture is in use.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateTextureFromMetal(
    device: native::WGPUDevice,
    mtl_texture: *mut std::ffi::c_void,
    descriptor: Option<&native::WGPUTextureDescriptor>,
) -> native::WGPUTexture {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let descriptor = descriptor.expect("invalid descriptor");
    let Some(format) = map_texture_format(descriptor.format) else {
        eprintln!("ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: an invalid format");
        return std::ptr::null_mut();
    };
    let usage = crate::conv::map_texture_usage_flags(descriptor.usage);
    if descriptor.dimension != native::WGPUTextureDimension_2D
        || descriptor.size.depthOrArrayLayers != 1
        || descriptor.mipLevelCount != 1
        || descriptor.sampleCount != 1
    {
        eprintln!(
            "ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: only a two-dimensional texture of one layer, level and sample"
        );
        return std::ptr::null_mut();
    }
    let Some(raw) = Retained::retain(mtl_texture.cast::<ProtocolObject<dyn MTLTexture>>()) else {
        eprintln!("ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: a null Metal texture");
        return std::ptr::null_mut();
    };
    let hal_texture = hal::metal::Device::texture_from_raw(
        raw,
        format,
        MTLTextureType::Type2D,
        1,
        1,
        hal::CopyExtent { width: descriptor.size.width, height: descriptor.size.height, depth: 1 },
        None,
    );
    let desc = wgt::TextureDescriptor {
        label: crate::utils::string_view_into_label(descriptor.label),
        size: crate::conv::map_extent3d(&descriptor.size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgt::TextureDimension::D2,
        format,
        usage,
        view_formats: Vec::new(),
    };
    let (texture_id, error) = context.create_texture_from_hal(
        Box::new(hal_texture),
        device_id,
        &desc,
        wgt::TextureUses::UNINITIALIZED,
        None,
    );
    if let Some(cause) = error {
        crate::handle_error(error_sink, cause, desc.label, "wgpuDeviceCreateTextureFromMetal");
    }
    Arc::into_raw(Arc::new(WGPUTextureImpl {
        context: context.clone(),
        id: texture_id,
        error_sink: error_sink.clone(),
        surface_id: None,
        has_surface_presented: Arc::new(atomic::AtomicBool::default()),
        data: TextureData {
            usage: descriptor.usage,
            dimension: descriptor.dimension,
            size: descriptor.size,
            format: descriptor.format,
            mip_level_count: 1,
            sample_count: 1,
        },
    }))
}
