use std::io::{prelude::*, BufReader};
use std::fs::File;
use inference_server::nn::nn::LinearLayer;
use inference_server::server::server::SimpleServer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut fin = File::open("lin.bin")?;
    let mut data: Vec<u8> = vec![];
    fin.read_to_end(&mut data)?;
    let mut v: Vec<f32> = vec![];
    for chunk in data.chunks(4){
        let chunk: [u8; 4] = chunk.try_into()?;
        v.push(f32::from_le_bytes(chunk));
    }
    let model = LinearLayer::from_vec(v);
    let server = SimpleServer::new("127.0.0.1", "7878", model);
    server.serve_forever();

    return Ok(());
}
