pub mod server {

    use std::{
        io::{prelude::*, BufReader},
        net::{TcpListener, TcpStream},
    };
    use crate::nn::nn::LinearLayer;

    pub struct SimpleServer {
        addr: String,
        port: String,
        listener: TcpListener,
        model: LinearLayer, 
    }

    impl SimpleServer {

        pub fn new(addr: &str, port: &str, model: LinearLayer) -> Self {
            let addr_copy = addr.clone();
            let full_addr = format!("{}:{}", addr, port);
            return Self {
                addr: addr.to_owned(),
                port: port.to_string(),
                listener: TcpListener::bind(full_addr).unwrap(),
                model
            };
        }
        // pub fn new(addr: String, port: String, model: LinearLayer) -> Self {
        //     let addr_copy = addr.clone();
        //     let full_addr: String = addr_copy + ":" + &port;
        //     return Self {
        //         addr: addr.clone(),
        //         port: port.clone(),
        //         listener: TcpListener::bind(full_addr).unwrap(),
        //         model
        //     };
        // }

        pub fn send_error(&self, mut stream: TcpStream, msg: String) {
            let response = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{}", msg);
            stream.write_all(response.as_bytes());
        }

        pub fn handle_connection(&self, mut stream: TcpStream) {
            println!("Got a new connection!");
            let buf_reader = BufReader::new( &mut stream);
            let http_request: Vec<_> = buf_reader
                .lines()
                .map(|result| result.unwrap())
                .take_while(|line| !line.contains("EOS"))
                .collect();

            let data = &http_request[http_request.len() - 1];
            let numbers: Vec<f32> = data.split(" ").map(|x| x.parse::<f32>().unwrap()).collect();
            if numbers.len() != self.model.size() {
                self.send_error(stream, format!("Size mismatch: expected {} got {}", self.model.size(), numbers.len()));
                return;
            }
            let result = self.model.forward(numbers);
            let response = format!("HTTP/1.1 200 OK\r\n\r\n{}", result);
            dbg!(http_request);
            stream.write_all(response.as_bytes());
        }

        pub fn serve_forever(&self) {
            println!("Start serving at {}:{}", self.addr, self.port);
            for stream in self.listener.incoming() {
                let stream = stream.unwrap();
                self.handle_connection(stream);
            }
        }
    }
}