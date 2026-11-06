//! Native-object interop for hosts that own the native GPU objects (the ProGPU engine in FerroUI,
//! branch `progpu/interop` of wieslawsoltes/wgpu-native).
//!
//! A window platform that presents its own frames (a `CAMetalLayer` whose drawables it presents
//! inside the window's Core Animation transaction, so a resizing window never shows a stretched
//! frame) hands its renderer the Metal device, the command queue and each frame's texture. These
//! functions make a wgpu device on that device and queue, and a wgpu texture of such a texture,
//! through wgpu-hal: the work of the device is submitted to the platform's queue, in order with the
//! platform's own work, so the frame needs no cross-device synchronization.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::conv::map_device_descriptor;
use crate::native;
use crate::utils::get_base_device_limits_from_adapter_limits;
use crate::{
    follow_chain, ptr_into_label, ptr_into_path, ErrorSinkRaw, QueueId, TextureData, WGPUAdapterImpl, WGPUDeviceImpl,
    WGPUTextureImpl, DEFAULT_DEVICE_LOST_HANDLER,
};
use foreign_types::ForeignTypeRef;
use parking_lot::Mutex;
use std::sync::{atomic, Arc, OnceLock};

/// Creates a device of the Metal adapter of `instance` on the host's `id<MTLDevice>` and
/// `id<MTLCommandQueue>` (both retained by the device until it is released). The descriptor is
/// that of `wgpuAdapterRequestDevice` (required features and limits, the device-lost callback;
/// null for the defaults). Returns null when the instance has no Metal adapter or the device
/// cannot be created; the reason is printed to the standard error.
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
    if mtl_device.is_null() || mtl_queue.is_null() {
        eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: a null Metal device or queue");
        return std::ptr::null_mut();
    }
    let adapter_id = match context.request_adapter(
        &wgt::RequestAdapterOptions::default(),
        wgc::instance::AdapterInputs::Mask(wgt::Backends::METAL, |_| ()),
    ) {
        Ok(id) => id,
        Err(error) => {
            eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: no Metal adapter: {error}");
            return std::ptr::null_mut();
        }
    };
    // Dropped after the device is created (the device keeps what it needs of the adapter).
    let adapter = Arc::new(WGPUAdapterImpl { context: context.clone(), id: adapter_id, properties: OnceLock::default() });
    let adapter_limits = match context.adapter_limits::<wgc::api::Metal>(adapter_id) {
        Ok(limits) => limits,
        Err(error) => {
            eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: {error}");
            return std::ptr::null_mut();
        }
    };
    let base_limits = get_base_device_limits_from_adapter_limits(&adapter_limits);
    let (desc, trace_str, device_lost_handler) = match descriptor {
        Some(descriptor) => {
            let (desc, trace_str) = follow_chain!(
                map_device_descriptor((descriptor, base_limits),
                WGPUSType_DeviceExtras => native::WGPUDeviceExtras)
            );
            let handler = crate::DeviceLostCallback {
                callback: descriptor.deviceLostCallback,
                userdata: descriptor.deviceLostUserdata,
            };
            (desc, trace_str, handler)
        }
        None => (
            wgt::DeviceDescriptor { required_limits: base_limits, ..Default::default() },
            std::ptr::null(),
            DEFAULT_DEVICE_LOST_HANDLER,
        ),
    };
    // Retained here; the hal objects own one reference each.
    let raw_device = metal::DeviceRef::from_ptr(mtl_device.cast()).to_owned();
    let raw_queue = metal::CommandQueueRef::from_ptr(mtl_queue.cast()).to_owned();
    let open = hal::OpenDevice {
        device: hal::metal::Device::device_from_raw(raw_device, desc.required_features),
        queue: hal::metal::Queue::queue_from_raw(raw_queue, 1.0),
    };
    let (device_id, queue_id, error) = context.create_device_from_hal::<wgc::api::Metal>(
        adapter_id,
        open,
        &desc,
        ptr_into_path(trace_str),
        (),
        (),
    );
    drop(adapter);
    if let Some(error) = error {
        eprintln!("ERROR: [wgpu-native]: wgpuInstanceCreateDeviceFromMetal: {error}");
        return std::ptr::null_mut();
    }
    Arc::into_raw(Arc::new(WGPUDeviceImpl {
        context: context.clone(),
        id: device_id,
        queue: Arc::new(QueueId { context: context.clone(), id: queue_id }),
        error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new(device_lost_handler))),
    }))
}

/// Creates a texture of the host's two-dimensional `id<MTLTexture>` (retained by the texture until
/// it is released) on a device made by [`wgpuInstanceCreateDeviceFromMetal`] on the texture's
/// device. The descriptor states what the Metal texture is: its size (depth one), format, usage,
/// one mip level and one sample. Returns null when the descriptor does not describe a texture that
/// wgpu can wrap.
///
/// # Safety
/// `mtl_texture` is a live `id<MTLTexture>` of the device's Metal device whose properties are the
/// descriptor's; it must not be destroyed by its owner while the texture is used.
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
    let Some(format) = crate::conv::map_texture_format(descriptor.format) else {
        eprintln!("ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: an invalid format");
        return std::ptr::null_mut();
    };
    let Some(usage) = wgt::TextureUsages::from_bits(descriptor.usage) else {
        eprintln!("ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: an invalid usage");
        return std::ptr::null_mut();
    };
    if mtl_texture.is_null()
        || descriptor.dimension != native::WGPUTextureDimension_2D
        || descriptor.size.depthOrArrayLayers != 1
        || descriptor.mipLevelCount != 1
        || descriptor.sampleCount != 1
    {
        eprintln!("ERROR: [wgpu-native]: wgpuDeviceCreateTextureFromMetal: only a two-dimensional texture of one layer, level and sample");
        return std::ptr::null_mut();
    }
    let raw = metal::TextureRef::from_ptr(mtl_texture.cast()).to_owned();
    let hal_texture = hal::metal::Device::texture_from_raw(
        raw,
        format,
        metal::MTLTextureType::D2,
        1,
        1,
        hal::CopyExtent { width: descriptor.size.width, height: descriptor.size.height, depth: 1 },
    );
    let desc = wgt::TextureDescriptor {
        label: ptr_into_label(descriptor.label),
        size: crate::conv::map_extent3d(&descriptor.size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgt::TextureDimension::D2,
        format,
        usage,
        view_formats: Vec::new(),
    };
    let (texture_id, error) = context.create_texture_from_hal::<wgc::api::Metal>(hal_texture, device_id, &desc, ());
    if let Some(cause) = error {
        crate::handle_error(context, error_sink, cause, "", desc.label, "wgpuDeviceCreateTextureFromMetal");
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
