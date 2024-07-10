use std::{
    io::{prelude::*, BufReader},
    net::{TcpListener, TcpStream},
    sync::mpsc::{channel, Receiver, Sender},
    sync::Arc,
    time::{Duration, Instant},
};

use threadpool::ThreadPool;

use crate::nn::LinearLayer;

pub struct SimpleServer {
    addr: String,
    port: String,
    listener: TcpListener,
    model: Arc<LinearLayer>,
    num_threads: usize,
    pool: ThreadPool,
}

impl SimpleServer {
    pub fn new(addr: &str, port: &str, model: LinearLayer) -> Self {
        let full_addr = format!("{}:{}", addr, port);
        let num_threads: usize = 4;
        let pool = ThreadPool::new(num_threads);
        return Self {
            addr: addr.to_owned(),
            port: port.to_string(),
            listener: TcpListener::bind(full_addr).unwrap(),
            model: Arc::new(model),
            num_threads,
            pool,
        };
    }

    pub fn send_error(&self, mut stream: TcpStream, msg: String) {
        let response = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{}", msg);
        let _ = stream.write_all(response.as_bytes());
    }

    pub fn handle_connection(&self, mut stream: TcpStream) {
        let buf_reader = BufReader::new(&mut stream);
        let http_request: Vec<_> = buf_reader
            .lines()
            .map(|result| result.unwrap())
            .take_while(|line| !line.contains("EOS"))
            .collect();

        let data = &http_request[http_request.len() - 1];
        let numbers: Vec<f32> = data.split(" ").map(|x| x.parse::<f32>().unwrap()).collect();
        let numbers: Vec<Vec<f32>> = vec![numbers];
        let result = self.model.forward(&numbers);
        // let result = self.model.forward_par(numbers);
        let response = format!("HTTP/1.1 200 OK\r\n\r\n{}", result[0][0]);
        let _ = stream.write_all(response.as_bytes());
    }

    fn model_batched_inference(
        model: Arc<LinearLayer>,
        rx: Receiver<(Vec<f32>, Sender<Vec<f32>>)>,
    ) {
        let time_freq = Duration::from_secs(1);
        loop {
            let start_time = Instant::now();
            let mut batch: Vec<Vec<f32>> = vec![];
            let mut send_channels: Vec<Sender<Vec<f32>>> = vec![];
            while batch.len() < 8 {
                if batch.len() > 0 && Instant::now() - start_time > time_freq {
                    break;
                }
                if let Ok((x, tx)) = rx.recv() {
                    batch.push(x);
                    send_channels.push(tx);
                }
            }
            let results: Vec<Vec<f32>> = model.forward(&batch);
            for (result, channel) in results.into_iter().zip(send_channels.iter()) {
                channel.send(result);
            }
        }
    }

    fn echo() {
        println!("Echo");
    }

    pub fn serve_forever(&self) {
        println!("Start serving at {}:{}", self.addr, self.port);
        for stream in self.listener.incoming() {
            let stream = stream.unwrap();
            self.pool.execute(|| SimpleServer::echo());
            self.handle_connection(stream);
        }
    }
}
