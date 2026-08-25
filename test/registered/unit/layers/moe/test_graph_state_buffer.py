import torch

from sglang.srt.mem_cache.graph_state_buffer import GraphStateBuffer


def test_graph_state_buffer_keeps_fixed_addresses_and_local_indices():
    buffer = GraphStateBuffer(
        conv_shapes=[(1, 3)],
        temporal_shape=(2, 4),
        max_batch_size=4,
        conv_dtype=torch.float32,
        temporal_dtype=torch.float32,
        device="cpu",
    )
    conv_ptr = buffer.conv[0].data_ptr()
    temporal_ptr = buffer.temporal.data_ptr()
    batch = buffer.load(
        [torch.ones(1, 2, 3)], torch.full((2, 2, 4), 2.0)
    )
    again = buffer.load(
        [torch.full((1, 1, 3), 3.0)], torch.full((2, 1, 4), 4.0)
    )
    assert buffer.conv[0].data_ptr() == conv_ptr
    assert buffer.temporal.data_ptr() == temporal_ptr
    assert batch.kernel_indices.tolist() == [0, 1]
    assert again.kernel_indices.tolist() == [0]
    assert again.conv[0].flatten().tolist() == [3.0] * 3
