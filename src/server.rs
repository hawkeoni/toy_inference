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
                .take_while(|line| !line.is_empty())
                .collect();

            stream.write_all("hello_there".as_bytes());
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