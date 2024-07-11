use std::io::{prelude::*, BufReader};
use std::fs::File;
use inference_server::nn::{LinearLayer, InferenceMode};
use inference_server::server::SimpleServer;
use inference_server::nn::utils::create_random_matrix;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mut fin = File::open("lin.bin")?;
    // let mut data: Vec<u8> = vec![];
    // fin.read_to_end(&mut data)?;
    // let mut v: Vec<f32> = vec![];
    // for chunk in data.chunks(4){
    //     let chunk: [u8; 4] = chunk.try_into()?;
    //     v.push(f32::from_le_bytes(chunk));
    // }
    // let layer = LinearLayer::new(create_random_matrix(1024, 2048), InferenceMode::Naive);
    let layer = LinearLayer::new(create_random_matrix(1024, 2048), InferenceMode::Rayon);
    let server = SimpleServer::new("127.0.0.1", "7878", layer);
    // server.serve_forever_simple();
    server.serve_forever_threads(20);
    // println!("Hello from main");

    return Ok(());
}
