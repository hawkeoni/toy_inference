use inference_server::nn::{LinearLayer, InferenceMode};
use inference_server::server::SimpleServer;
use inference_server::server_grpc::GrpcServer;
use inference_server::nn::utils::create_random_matrix;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layer = LinearLayer::new(
        create_random_matrix(1024, 2048), 
        // InferenceMode::Rayon
        InferenceMode::Gpu
    );
    // let server = SimpleServer::new("127.0.0.1", "7878", layer);
    let server = GrpcServer::new("127.0.0.1", "7878", layer);
    // server.serve_forever_simple();
    server.serve().await;
    return Ok(());
}
