mod server;
use server::server::SimpleServer;

fn main() {
    let server = SimpleServer::new("127.0.0.1".to_string(), "7878".to_string());
    server.serve_forever();
}
