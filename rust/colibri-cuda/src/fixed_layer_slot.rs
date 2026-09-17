use super::*;

/// 固定层设备权重的跨层复用槽位。
///
/// 原状是每层每 token 都重建一遍 `LayerDeviceWeights`：约 20 次 `cuMemAlloc`
/// 加 20 次 `cuMemFree`。2026-09-17 的引擎内插桩量到这两段占 `fixed` 的一半
/// （`414 ms/token`），而真正的拷贝只占 51%。这里只做一件事：把上一层的设备
/// 缓冲留着，下一层把宿主负载拷进去（restage），不再重新申请。
///
/// 字节账不变——restage 拷的是同一批字节；变的是不再有分配/释放，也不再有它们
/// 带出来的缺页。所以判这一层改动的证据是 `fixed` 的墙和输出 digest，不是
/// `fixed_h2d_bytes` 那个按缓冲尺寸算出来的数。
///
/// 按层型分三个槽位：Csa4 层带 indexer、Hca128 层不带，两者在层间交替，混在一个
/// 槽位里永远匹配不上（第一版就是这么写错、每 token 只复用了一次）。
///
/// 槽位由调用方持有（Flash B1 请求驱动），与 `host_cache` 同寿命；restage 覆盖
/// `new_layer_with_wo_a` 会填的全部张量，长度不一致直接报错，不静默读旧数据。
pub(crate) struct FixedLayerSlot {
    slots: [Option<LayerDeviceWeights>; 3],
    spare_indexer: Option<DeviceLayerIndexer>,
}

impl FixedLayerSlot {
    pub(crate) fn new() -> Self {
        Self {
            slots: [None, None, None],
            spare_indexer: None,
        }
    }

    fn kind(ratio: v4_state::CompressionRatio, has_indexer: bool) -> usize {
        match (ratio, has_indexer) {
            (v4_state::CompressionRatio::WindowOnly, _) => 0,
            (v4_state::CompressionRatio::Hca128, _) => 1,
            (v4_state::CompressionRatio::Csa4, _) => 2,
        }
    }

    /// 交出复用权重；这一层型的槽位还空着就返回 None，调用方去新建。
    pub(crate) fn take_restaged(
        &mut self,
        stream: &CudaStream,
        host: &DecodedLayerHost,
    ) -> Result<Option<LayerDeviceWeights>, Box<dyn Error>> {
        let index = Self::kind(host.state.compression_ratio, host.state.indexer.is_some());
        let Some(mut weights) = self.slots[index].take() else {
            return Ok(None);
        };
        weights.restage_from_host(
            stream,
            &host.state,
            &host.projections,
            &host.tail,
            &mut self.spare_indexer,
        )?;
        Ok(Some(weights))
    }

    /// 一层跑完把权重还回它层型对应的槽位。
    pub(crate) fn release(&mut self, weights: LayerDeviceWeights) {
        let index = Self::kind(weights.compression_ratio, weights.indexer.is_some());
        self.slots[index] = Some(weights);
    }
}
