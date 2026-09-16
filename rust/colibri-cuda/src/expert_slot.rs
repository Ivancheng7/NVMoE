use super::*;

pub(crate) struct ExpertSlot<'a> {
    pub(crate) spec: &'static v4_spec::V4Spec,
    pub(crate) fp4_accum_arm: FlashFp4AccumArm,
    pub(crate) tilelang_ref: Option<v4_tilelang_ref::TilelangFp4Reference>,
    pub(crate) module: &'a kernels::LoadedModule,
    pub(crate) stream: std::sync::Arc<CudaStream>,
    pub(crate) w1_host: PinnedHostBuffer<u8>,
    pub(crate) w1_scale_host: PinnedHostBuffer<u8>,
    pub(crate) w2_host: PinnedHostBuffer<u8>,
    pub(crate) w2_scale_host: PinnedHostBuffer<u8>,
    pub(crate) w3_host: PinnedHostBuffer<u8>,
    pub(crate) w3_scale_host: PinnedHostBuffer<u8>,
    pub(crate) input_host: PinnedHostBuffer<u16>,
    pub(crate) output_host: PinnedHostBuffer<f32>,
    pub(crate) input_device: DeviceBuffer<u16>,
    pub(crate) w1_device: DeviceBuffer<u8>,
    pub(crate) w1_scale_device: DeviceBuffer<u8>,
    pub(crate) w2_device: DeviceBuffer<u8>,
    pub(crate) w2_scale_device: DeviceBuffer<u8>,
    pub(crate) w3_device: DeviceBuffer<u8>,
    pub(crate) w3_scale_device: DeviceBuffer<u8>,
    pub(crate) input_quantized: DeviceBuffer<f32>,
    pub(crate) input_fp8: Option<DeviceBuffer<u8>>,
    pub(crate) input_fp8_scales: Option<DeviceBuffer<u8>>,
    pub(crate) gate: DeviceBuffer<u16>,
    pub(crate) up: DeviceBuffer<u16>,
    pub(crate) routed: DeviceBuffer<u16>,
    pub(crate) routed_quantized: DeviceBuffer<f32>,
    pub(crate) routed_fp8: Option<DeviceBuffer<u8>>,
    pub(crate) routed_fp8_scales: Option<DeviceBuffer<u8>>,
    pub(crate) output_bf16: DeviceBuffer<u16>,
    pub(crate) output: DeviceBuffer<f32>,
    pub(crate) weight_h2d_ms: f64,
    pub(crate) weight_start: cuda_core::CudaEvent,
    pub(crate) weight_end: cuda_core::CudaEvent,
    pub(crate) kernel_start: cuda_core::CudaEvent,
    pub(crate) kernel_end: cuda_core::CudaEvent,
    pub(crate) complete: cuda_core::CudaEvent,
}

impl<'a> ExpertSlot<'a> {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        weights: &v4_expert::RoutedExpert,
    ) -> Result<Self, Box<dyn Error>> {
        // ponytail: one L0 expert at a time; add slot management only after
        // measured multi-expert traffic requires it.
        let w1_host = PinnedHostBuffer::from_slice(context, &weights.w1)?;
        let w1_scale_host = PinnedHostBuffer::from_slice(context, &weights.w1_scale)?;
        let w2_host = PinnedHostBuffer::from_slice(context, &weights.w2)?;
        let w2_scale_host = PinnedHostBuffer::from_slice(context, &weights.w2_scale)?;
        let w3_host = PinnedHostBuffer::from_slice(context, &weights.w3)?;
        let w3_scale_host = PinnedHostBuffer::from_slice(context, &weights.w3_scale)?;
        let spec = weights.spec;
        let input_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let output_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let input_device = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let mut w1_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w1.len())? };
        let mut w1_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w1_scale.len())? };
        let mut w2_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w2.len())? };
        let mut w2_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w2_scale.len())? };
        let mut w3_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w3.len())? };
        let mut w3_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w3_scale.len())? };
        let weight_h2d_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_h2d_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        weight_h2d_start.record(stream)?;
        // SAFETY: pinned weight sources live through the synchronized completion
        // event below; every destination has the exact matching allocation.
        unsafe {
            w1_device.copy_from_pinned_host_async(stream, &w1_host)?;
            w1_scale_device.copy_from_pinned_host_async(stream, &w1_scale_host)?;
            w2_device.copy_from_pinned_host_async(stream, &w2_host)?;
            w2_scale_device.copy_from_pinned_host_async(stream, &w2_scale_host)?;
            w3_device.copy_from_pinned_host_async(stream, &w3_host)?;
            w3_scale_device.copy_from_pinned_host_async(stream, &w3_scale_host)?;
        }
        weight_h2d_end.record(stream)?;
        weight_h2d_end.synchronize()?;
        let weight_h2d_ms = weight_h2d_start.elapsed_ms(&weight_h2d_end)? as f64;
        let fp4_accum_arm = flash_fp4_accum_arm(spec)?;
        let tilelang_ref = if fp4_accum_arm == FlashFp4AccumArm::TilelangRef {
            let model_dir = v4_model_dir_for(spec)?;
            let reference_dir = v4_tilelang_ref::reference_dir_from_env()?;
            Some(v4_tilelang_ref::load_for_flash(
                context,
                spec,
                &model_dir,
                &reference_dir,
            )?)
        } else {
            None
        };
        let input_quantized = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let (input_fp8, input_fp8_scales, routed_fp8, routed_fp8_scales) =
            if fp4_accum_arm.uses_raw_fp8() {
                (
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.hidden)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.hidden / v4_state::FP8_BLOCK,
                    )?),
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.moe_intermediate)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.moe_intermediate / v4_state::FP8_BLOCK,
                    )?),
                )
            } else {
                (None, None, None, None)
            };
        let gate = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let up = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed_quantized =
            unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let output_bf16 = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let output = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let kernel_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let kernel_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let complete = context.new_event(None)?;
        complete.record(stream)?;
        complete.synchronize()?;
        Ok(Self {
            spec,
            fp4_accum_arm,
            tilelang_ref,
            module,
            stream: stream.clone(),
            w1_host,
            w1_scale_host,
            w2_host,
            w2_scale_host,
            w3_host,
            w3_scale_host,
            input_host,
            output_host,
            input_device,
            w1_device,
            w1_scale_device,
            w2_device,
            w2_scale_device,
            w3_device,
            w3_scale_device,
            input_quantized,
            input_fp8,
            input_fp8_scales,
            gate,
            up,
            routed,
            routed_quantized,
            routed_fp8,
            routed_fp8_scales,
            output_bf16,
            output,
            weight_h2d_ms,
            weight_start: weight_h2d_start,
            weight_end: weight_h2d_end,
            kernel_start,
            kernel_end,
            complete,
        })
    }

    pub(crate) fn new_empty(
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new_empty_for(&v4_spec::PRO, context, module, stream)
    }

    pub(crate) fn new_empty_for(
        spec: &'static v4_spec::V4Spec,
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
    ) -> Result<Self, Box<dyn Error>> {
        let w1_bytes = spec.moe_intermediate * spec.hidden / 2;
        let w1_scale_bytes = spec.moe_intermediate * spec.hidden / 32;
        let w2_bytes = spec.hidden * spec.moe_intermediate / 2;
        let w2_scale_bytes = spec.hidden * spec.moe_intermediate / 32;
        let w1_host = PinnedHostBuffer::zeroed(context, w1_bytes)?;
        let w1_scale_host = PinnedHostBuffer::zeroed(context, w1_scale_bytes)?;
        let w2_host = PinnedHostBuffer::zeroed(context, w2_bytes)?;
        let w2_scale_host = PinnedHostBuffer::zeroed(context, w2_scale_bytes)?;
        let w3_host = PinnedHostBuffer::zeroed(context, w1_bytes)?;
        let w3_scale_host = PinnedHostBuffer::zeroed(context, w1_scale_bytes)?;
        let input_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let output_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let input_device = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let w1_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_bytes)? };
        let w1_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_scale_bytes)? };
        let w2_device = unsafe { DeviceBuffer::uninitialized_async(stream, w2_bytes)? };
        let w2_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w2_scale_bytes)? };
        let w3_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_bytes)? };
        let w3_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_scale_bytes)? };
        let fp4_accum_arm = flash_fp4_accum_arm(spec)?;
        let tilelang_ref = if fp4_accum_arm == FlashFp4AccumArm::TilelangRef {
            let model_dir = v4_model_dir_for(spec)?;
            let reference_dir = v4_tilelang_ref::reference_dir_from_env()?;
            Some(v4_tilelang_ref::load_for_flash(
                context,
                spec,
                &model_dir,
                &reference_dir,
            )?)
        } else {
            None
        };
        let input_quantized = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let (input_fp8, input_fp8_scales, routed_fp8, routed_fp8_scales) =
            if fp4_accum_arm.uses_raw_fp8() {
                (
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.hidden)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.hidden / v4_state::FP8_BLOCK,
                    )?),
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.moe_intermediate)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.moe_intermediate / v4_state::FP8_BLOCK,
                    )?),
                )
            } else {
                (None, None, None, None)
            };
        let gate = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let up = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed_quantized =
            unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let output_bf16 = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let output = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let kernel_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let kernel_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let complete = context.new_event(None)?;
        complete.record(stream)?;
        complete.synchronize()?;
        Ok(Self {
            spec,
            fp4_accum_arm,
            tilelang_ref,
            module,
            stream: stream.clone(),
            w1_host,
            w1_scale_host,
            w2_host,
            w2_scale_host,
            w3_host,
            w3_scale_host,
            input_host,
            output_host,
            input_device,
            w1_device,
            w1_scale_device,
            w2_device,
            w2_scale_device,
            w3_device,
            w3_scale_device,
            input_quantized,
            input_fp8,
            input_fp8_scales,
            gate,
            up,
            routed,
            routed_quantized,
            routed_fp8,
            routed_fp8_scales,
            output_bf16,
            output,
            weight_h2d_ms: 0.0,
            weight_start,
            weight_end,
            kernel_start,
            kernel_end,
            complete,
        })
    }

    pub(crate) fn load_range_execute_into(
        &mut self,
        shard: &safetensors::ShardFile,
        plan: &v4_expert::RoutedExpertRangePlan,
        input: &[u16],
        route_weight: f32,
        output: &mut [f32],
        profile_origin: Option<std::time::Instant>,
        profile_submit_ns: Option<u64>,
    ) -> Result<RoutedLaneExecution, Box<dyn Error>> {
        let mut timeline = profile_origin.map(|origin| RoutedLaneTimeline {
            submit_ns: profile_submit_ns.unwrap_or_else(|| timeline_elapsed_ns(origin)),
            read_start_ns: timeline_elapsed_ns(origin),
            read_end_ns: 0,
            h2d_enqueued_ns: 0,
            kernel_enqueued_ns: 0,
            d2h_enqueued_ns: 0,
            complete_ns: 0,
            h2d_event_ns: 0,
            kernel_event_ns: 0,
        });
        let read_started = std::time::Instant::now();
        plan.read_into(
            shard,
            [
                self.w1_host.as_mut_slice(),
                self.w1_scale_host.as_mut_slice(),
                self.w2_host.as_mut_slice(),
                self.w2_scale_host.as_mut_slice(),
                self.w3_host.as_mut_slice(),
                self.w3_scale_host.as_mut_slice(),
            ],
        )?;
        let read_ns = u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Some(timeline) = timeline.as_mut() {
            timeline.read_end_ns = timeline_elapsed_ns(profile_origin.expect("timeline origin"));
        }
        self.weight_start.record(&self.stream)?;
        unsafe {
            self.w1_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_host)?;
            self.w1_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_scale_host)?;
            self.w2_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_host)?;
            self.w2_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_scale_host)?;
            self.w3_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_host)?;
            self.w3_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_scale_host)?;
        }
        if let Some(timeline) = timeline.as_mut() {
            timeline.h2d_enqueued_ns =
                timeline_elapsed_ns(profile_origin.expect("timeline origin"));
        }
        self.weight_end.record(&self.stream)?;
        let kernel_ns = self.execute_into(
            input,
            route_weight,
            output,
            timeline.as_mut(),
            profile_origin,
        )?;
        self.weight_end.synchronize()?;
        let h2d_ns = (self.weight_start.elapsed_ms(&self.weight_end)? as f64 * 1_000_000.0) as u64;
        if let Some(timeline) = timeline.as_mut() {
            timeline.h2d_event_ns = h2d_ns;
            timeline.kernel_event_ns = kernel_ns;
        }
        Ok(RoutedLaneExecution {
            read_ns,
            h2d_ns,
            kernel_ns,
            timeline,
        })
    }

    pub(crate) fn execute(
        &mut self,
        input: &[u16],
        route_weight: f32,
    ) -> Result<(Vec<f32>, u64), Box<dyn Error>> {
        let mut output = vec![0.0_f32; self.spec.hidden];
        let gpu_ns = self.execute_into(input, route_weight, &mut output, None, None)?;
        Ok((output, gpu_ns))
    }

    pub(crate) fn execute_with_flash_trace(
        &mut self,
        input: &[u16],
        route_weight: f32,
    ) -> Result<(Vec<f32>, u64, FlashExpertTrace), Box<dyn Error>> {
        let (output, gpu_ns) = self.execute(input, route_weight)?;
        let ((input_fp8, input_scale), (w2_input_fp8, w2_input_scale)) = if self
            .fp4_accum_arm
            .uses_raw_fp8()
        {
            let input_fp8 = self
                .input_fp8
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its input FP8 buffer")?;
            let input_scale = self
                .input_fp8_scales
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its input FP8 scales")?;
            let w2_input_fp8 = self
                .routed_fp8
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its routed FP8 buffer")?;
            let w2_input_scale = self
                .routed_fp8_scales
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its routed FP8 scales")?;
            (
                (
                    input_fp8.to_host_vec(&self.stream)?,
                    input_scale.to_host_vec(&self.stream)?,
                ),
                (
                    w2_input_fp8.to_host_vec(&self.stream)?,
                    w2_input_scale.to_host_vec(&self.stream)?,
                ),
            )
        } else {
            (
                flash_parity_capture_activation_fp8(self.module, &self.stream, &self.input_device)?,
                flash_parity_capture_activation_fp8(self.module, &self.stream, &self.routed)?,
            )
        };
        Ok((
            output,
            gpu_ns,
            FlashExpertTrace {
                input_fp8,
                input_scale,
                w1: self.gate.to_host_vec(&self.stream)?,
                w3: self.up.to_host_vec(&self.stream)?,
                swiglu: self.routed.to_host_vec(&self.stream)?,
                w2_input_fp8,
                w2_input_scale,
                output: self.output_bf16.to_host_vec(&self.stream)?,
            },
        ))
    }

    pub(crate) fn execute_into(
        &mut self,
        input: &[u16],
        route_weight: f32,
        output: &mut [f32],
        mut timeline: Option<&mut RoutedLaneTimeline>,
        profile_origin: Option<std::time::Instant>,
    ) -> Result<u64, Box<dyn Error>> {
        if v4_expert::validate_input_for(self.spec, input, route_weight)? != 1 {
            return Err("resident expert slot accepts exactly one row".into());
        }
        if output.len() != self.spec.hidden {
            return Err("resident expert output requires one complete hidden row".into());
        }
        self.stream.context().bind_to_thread()?;
        self.input_host.as_mut_slice().copy_from_slice(input);
        // SAFETY: the input stager and all device buffers remain owned by this
        // slot until the reused completion event synchronizes below.
        unsafe {
            self.input_device
                .copy_from_pinned_host_async(&self.stream, &self.input_host)?;
        }
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.kernel_enqueued_ns = timeline_elapsed_ns(origin);
        }
        self.kernel_start.record(&self.stream)?;
        if self.fp4_accum_arm.uses_raw_fp8() {
            {
                let input_fp8 = self
                    .input_fp8
                    .as_mut()
                    .ok_or("Flash block32 arm lost its input FP8 buffer")?;
                let input_scales = self
                    .input_fp8_scales
                    .as_mut()
                    .ok_or("Flash block32 arm lost its input FP8 scales")?;
                launch_v4_activation_quantize_raw(
                    self.module,
                    &self.stream,
                    &self.input_device,
                    input_fp8,
                    input_scales,
                )?;
            }
            let input_fp8 = self
                .input_fp8
                .as_ref()
                .ok_or("Flash block32 arm lost its input FP8 buffer")?;
            let input_scales = self
                .input_fp8_scales
                .as_ref()
                .ok_or("Flash block32 arm lost its input FP8 scales")?;
            match self.fp4_accum_arm {
                FlashFp4AccumArm::Block32 => {
                    launch_v4_fp4_matvec_block32(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_matvec_block32(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::TilelangRef => {
                    let reference = self
                        .tilelang_ref
                        .as_ref()
                        .ok_or("tilelang_ref arm lost its validated reference")?;
                    launch_v4_fp4_tilelang_ref(
                        reference,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_tilelang_ref(
                        reference,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::Fp8Mma => {
                    launch_v4_fp4_matvec_mma(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_matvec_mma(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::Current => {
                    return Err("current FP4 arm entered raw-FP8 branch".into());
                }
            }
        } else {
            launch_v4_activation_quantize(
                self.module,
                &self.stream,
                &self.input_device,
                &mut self.input_quantized,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.input_quantized,
                &self.w1_device,
                &self.w1_scale_device,
                &mut self.gate,
                1,
                self.spec.moe_intermediate,
                self.spec.hidden,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.input_quantized,
                &self.w3_device,
                &self.w3_scale_device,
                &mut self.up,
                1,
                self.spec.moe_intermediate,
                self.spec.hidden,
            )?;
        }
        launch_v4_swiglu_route(
            self.module,
            &self.stream,
            &self.gate,
            &self.up,
            &mut self.routed,
            route_weight,
        )?;
        if self.fp4_accum_arm.uses_raw_fp8() {
            {
                let routed_fp8 = self
                    .routed_fp8
                    .as_mut()
                    .ok_or("Flash block32 arm lost its routed FP8 buffer")?;
                let routed_scales = self
                    .routed_fp8_scales
                    .as_mut()
                    .ok_or("Flash block32 arm lost its routed FP8 scales")?;
                launch_v4_activation_quantize_raw(
                    self.module,
                    &self.stream,
                    &self.routed,
                    routed_fp8,
                    routed_scales,
                )?;
            }
            let routed_fp8 = self
                .routed_fp8
                .as_ref()
                .ok_or("Flash block32 arm lost its routed FP8 buffer")?;
            let routed_scales = self
                .routed_fp8_scales
                .as_ref()
                .ok_or("Flash block32 arm lost its routed FP8 scales")?;
            match self.fp4_accum_arm {
                FlashFp4AccumArm::Block32 => launch_v4_fp4_matvec_block32(
                    self.module,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::TilelangRef => launch_v4_fp4_tilelang_ref(
                    self.tilelang_ref
                        .as_ref()
                        .ok_or("tilelang_ref arm lost its validated reference")?,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::Fp8Mma => launch_v4_fp4_matvec_mma(
                    self.module,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::Current => {
                    return Err("current FP4 arm entered raw-FP8 branch".into());
                }
            }
        } else {
            launch_v4_activation_quantize(
                self.module,
                &self.stream,
                &self.routed,
                &mut self.routed_quantized,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.routed_quantized,
                &self.w2_device,
                &self.w2_scale_device,
                &mut self.output_bf16,
                1,
                self.spec.hidden,
                self.spec.moe_intermediate,
            )?;
        }
        launch_v4_bf16_to_f32(
            self.module,
            &self.stream,
            &self.output_bf16,
            &mut self.output,
        )?;
        self.kernel_end.record(&self.stream)?;
        // SAFETY: output_host stays exclusively borrowed until `complete`
        // proves the D2H copy has finished.
        unsafe {
            self.output
                .copy_to_pinned_host_async(&self.stream, &mut self.output_host)?;
        }
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.d2h_enqueued_ns = timeline_elapsed_ns(origin);
        }
        self.complete.record(&self.stream)?;
        self.complete.synchronize()?;
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.complete_ns = timeline_elapsed_ns(origin);
        }
        let gpu_ns = (self.kernel_start.elapsed_ms(&self.kernel_end)? as f64 * 1_000_000.0) as u64;
        output.copy_from_slice(self.output_host.as_slice());
        Ok(gpu_ns)
    }

    pub(crate) fn copy_weight_tensors_from(
        &mut self,
        source: &Self,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn Error>> {
        self.w1_device
            .copy_from_device_async(&source.w1_device, stream)?;
        self.w1_scale_device
            .copy_from_device_async(&source.w1_scale_device, stream)?;
        self.w2_device
            .copy_from_device_async(&source.w2_device, stream)?;
        self.w2_scale_device
            .copy_from_device_async(&source.w2_scale_device, stream)?;
        self.w3_device
            .copy_from_device_async(&source.w3_device, stream)?;
        self.w3_scale_device
            .copy_from_device_async(&source.w3_scale_device, stream)?;
        Ok(())
    }

    pub(crate) fn finish_qd4_loan(&self) -> Result<bool, Box<dyn Error>> {
        self.complete.record(&self.stream)?;
        self.complete.synchronize()?;
        Ok(true)
    }
}
