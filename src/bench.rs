use inference_server::nn::{LinearLayer, InferenceMode};
use inference_server::nn::utils::{create_random_matrix, create_random_vector};
use std::time::{Instant, Duration};




fn bench_layer(layer: &LinearLayer, x: &Vec<Vec<f32>>, num_iters: u32, name: &str) {
    let start_time = Instant::now();
    for i in 0..num_iters {
        let res = layer.forward(x);
    }
    let passed_time = (Instant::now() - start_time).as_secs_f32();
    let time_per_iter = passed_time / (num_iters as f32);
    let (fan_in, fan_out) = layer.size();
    let div: i32 = 10;
    let div = div.pow(6) as f32;
    let flops = (x.len() * fan_in * fan_out) as f32 / time_per_iter / div;
    println!("Benchmarking {name} for {num_iters} iters took {passed_time}s, {}s per iter, MFLOPS {}", time_per_iter, flops);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim0 = 1024; let dim1 = 2048;
    let batch_size = 16;
    let w = create_random_matrix(dim0, dim1);
    let x = create_random_matrix(batch_size, dim0);
    let layer_naive = LinearLayer::new(w.clone(), InferenceMode::Naive);

    bench_layer(&layer_naive, &x, 10, "Naive inference");

    let layer_rayon = LinearLayer::new(w.clone(), InferenceMode::Rayon);
    bench_layer(&layer_rayon, &x, 10, "Rayon inference");

    return Ok(());
}