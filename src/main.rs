mod server;
mod nn;
use nn::nn::LinearLayer;
use server::server::SimpleServer;

fn main() {
    let model = LinearLayer::new(4.0, 5);
    let server = SimpleServer::new("127.0.0.1".to_string(), "7878".to_string(), model);
    server.serve_forever();
}
