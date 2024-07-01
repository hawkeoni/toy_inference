pub mod server {

    use std::{
        io::{prelude::*, BufReader},
        net::{TcpListener, TcpStream},
    };

    pub struct SimpleServer {
        addr: String,
        port: String,
        listener: TcpListener
    }

    impl SimpleServer {

        pub fn new(addr: String, port: String) -> Self {
            let addr_copy = addr.clone();
            let full_addr: String = addr_copy + ":" + &port;
            return Self {
                addr: addr.clone(),
                port: port.clone(),
                listener: TcpListener::bind(full_addr).unwrap()
            };
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
            let numbers: Vec<i32> = data.split(" ").map(|x| x.parse::<i32>().unwrap()).collect();
            let sum: i32 = numbers.iter().sum();
            let response = format!("HTTP/1.1 200 OK\r\n\r\n{}", sum);
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