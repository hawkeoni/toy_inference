use inference_server::nn::{LinearLayer, InferenceMode};
use inference_server::server::SimpleServer;
use inference_server::server_grpc::GrpcServer;
use inference_server::nn::utils::create_random_matrix;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size: Option<usize> = Some(8);
    let layer = LinearLayer::new(
        create_random_matrix(2048, 1024), 
        InferenceMode::Rayon,
        // InferenceMode::Gpu,
        batch_size.clone()
    );
    // let server = SimpleServer::new("127.0.0.1", "7878", layer);
    let server = GrpcServer::new("127.0.0.1", "7878", layer);
    // server.serve_forever_simple();
    server.serve(batch_size.clone()).await;
    return Ok(());
}
