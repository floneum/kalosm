use std::time::{Duration, Instant};

use fusor_core::{Device, Tensor};

async fn wait_for_submitted_work(device: &Device) {
    let (sender, receiver) = futures_channel::oneshot::channel();
    device.wgpu_queue().on_submitted_work_done(|| {
        _ = sender.send(());
    });
    let _ = receiver.await;
}

async fn measure_async_empty_submit(device: &Device) -> Duration {
    let encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("empty async overhead probe"),
        });
    let start = Instant::now();
    device.wgpu_queue().submit(Some(encoder.finish()));
    wait_for_submitted_work(device).await;
    start.elapsed()
}

fn measure_sync_empty_submit(device: &Device) -> Duration {
    let encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("empty sync overhead probe"),
        });
    let start = Instant::now();
    device.wgpu_queue().submit(Some(encoder.finish()));
    device.poll_wait();
    start.elapsed()
}

fn poll_until_empty(device: &Device) {
    loop {
        let status = device
            .wgpu_device()
            .poll(wgpu::PollType::Poll)
            .expect("failed to poll GPU device");
        if status.is_queue_empty() {
            break;
        }
        std::thread::yield_now();
    }
}

fn measure_poll_loop_empty_submit(device: &Device) -> Duration {
    let encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("empty poll-loop overhead probe"),
        });
    let start = Instant::now();
    device.wgpu_queue().submit(Some(encoder.finish()));
    poll_until_empty(device);
    start.elapsed()
}

async fn measure_callback_only(device: &Device) -> Duration {
    let start = Instant::now();
    wait_for_submitted_work(device).await;
    start.elapsed()
}

fn print_samples(label: &str, samples: &[Duration]) {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    eprintln!("{label}: median={median:?} min={min:?} max={max:?} samples={samples:?}");
}

#[tokio::main]
async fn main() {
    let repeats = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(20);

    let device = Device::new().await.unwrap();
    eprintln!("device={:?}", device.wgpu_adapter().get_info());

    let data = vec![vec![1.0f32; 100]; 100];
    let tensor = Tensor::new(&device, &data);
    _ = tensor.as_slice().await.unwrap();

    let add = tensor.clone() + 1.0;
    add.materialize().await;

    let mut callback_only = Vec::with_capacity(repeats);
    let mut empty_async = Vec::with_capacity(repeats);
    let mut empty_sync = Vec::with_capacity(repeats);
    let mut empty_poll_loop = Vec::with_capacity(repeats);
    let mut leaf_materialize = Vec::with_capacity(repeats);
    let mut add_materialize = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        callback_only.push(measure_callback_only(&device).await);
        empty_async.push(measure_async_empty_submit(&device).await);
        empty_sync.push(measure_sync_empty_submit(&device));
        empty_poll_loop.push(measure_poll_loop_empty_submit(&device));

        let start = Instant::now();
        tensor.materialize().await;
        leaf_materialize.push(start.elapsed());

        let add = tensor.clone() + 1.0;
        let start = Instant::now();
        add.materialize().await;
        add_materialize.push(start.elapsed());
    }

    print_samples("callback-only", &callback_only);
    print_samples("empty-submit-async", &empty_async);
    print_samples("empty-submit-sync", &empty_sync);
    print_samples("empty-submit-poll-loop", &empty_poll_loop);
    print_samples("leaf-materialize", &leaf_materialize);
    print_samples("add-materialize", &add_materialize);
}
