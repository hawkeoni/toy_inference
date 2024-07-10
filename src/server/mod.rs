use std::{
    io::{prelude::*, BufReader},
    net::{TcpListener, TcpStream},
    sync::mpsc::{channel, Receiver, Sender},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use std::thread;

use threadpool::ThreadPool;

use crate::nn::LinearLayer;

enum ServerInternalMessage {
    InferenceResponse(Vec<f32>),
    InferenceRequest((Vec<f32>, Sender<ServerInternalMessage>)),
}

pub struct SimpleServer {
    addr: String,
    port: String,
    listener: TcpListener,
    model: Arc<LinearLayer>,
    num_threads: usize,
    pool: ThreadPool,
    tx: Arc<Mutex<Sender<ServerInternalMessage>>>,
    rx: Arc<Mutex<Receiver<ServerInternalMessage>>>,
}


impl SimpleServer {
    pub fn new(addr: &str, port: &str, model: LinearLayer) -> Self {
        let (tx, rx) = channel::<ServerInternalMessage>();
        let full_addr = format!("{}:{}", addr, port);
        let num_threads: usize = 20;
        let pool = ThreadPool::new(num_threads);
        return Self {
            addr: addr.to_owned(),
            port: port.to_string(),
            listener: TcpListener::bind(full_addr).unwrap(),
            model: Arc::new(model),
            num_threads,
            pool,
            tx: Arc::new(Mutex::new(tx)),
            rx: Arc::new(Mutex::new(rx)),
        };
    }

    pub fn send_error(&self, mut stream: TcpStream, msg: String) {
        let response = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{}", msg);
        let _ = stream.write_all(response.as_bytes());
    }

    fn parse_payload_from_http(mut stream: &TcpStream) -> Vec<f32>{
        let mut empty_line_count = 0;
        let buf_reader = BufReader::new(&mut stream);
        let http_data: Vec<_> = buf_reader
            .lines()
            .map(|result| result.unwrap())
            .take_while(|line| {
                if line.trim().is_empty() {
                    empty_line_count += 1;
                    return empty_line_count < 2;
                }
                return true;
            })
            .collect();
        let numbers: Vec<f32> = http_data[http_data.len() - 1]
        .split(" ")
        .map(|x| x.parse::<f32>().unwrap())
        .collect();
        return numbers;
    }

    fn format_result_http(result: &Vec<f32>) -> String{
        let result_string = result
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(" ");
        let response = format!("HTTP/1.1 200 OK\r\n\r\n{}", result_string);
        return response;

    }

    pub fn handle_connection(&self, mut stream: TcpStream) {
        let payload = SimpleServer::parse_payload_from_http(&stream);
        let result = &self.model.forward(&vec![payload])[0];
        let http_response = SimpleServer::format_result_http(&result);
        let _ = stream.write_all(http_response.as_bytes()).unwrap();
    }

    fn inference_worker(
        model: Arc<LinearLayer>,
        inference_worker_receiver: Arc<Mutex<Receiver<ServerInternalMessage>>>,
    ) {
        let time_freq = Duration::from_millis(200);
        loop {
            let start_time = Instant::now();
            let mut batch: Vec<Vec<f32>> = vec![];
            let mut send_channels: Vec<Sender<ServerInternalMessage>> = vec![];
            while batch.len() < 8 {
                if batch.len() > 0 && Instant::now() - start_time > time_freq {
                    break;
                }
                let request = inference_worker_receiver.lock().unwrap().try_recv();
                match request {
                    Ok(ServerInternalMessage::InferenceRequest((x, tx))) => {
                        batch.push(x);
                        send_channels.push(tx);
                    },
                    Ok(ServerInternalMessage::InferenceResponse(_)) => panic!("We should not get response here."),
                    Err(err) => continue
                }
            }
            // dbg!(format!("Got a batch of {}", batch.len()));
            let results: Vec<Vec<f32>> = model.forward(&batch);
            for (result, channel) in results.into_iter().zip(send_channels.iter()) {
                let _ = channel.send(ServerInternalMessage::InferenceResponse(result));
            }
        }
    }

    fn connection_worker(mut stream: TcpStream, inference_worker_sender: Arc<Mutex<Sender<ServerInternalMessage>>>) {
        let (tx, rx) = channel::<ServerInternalMessage>();
        // let (tx: Sender<, rx: Receiver<Vec<f32>>) = channel();
        let payload = SimpleServer::parse_payload_from_http(&stream);
        inference_worker_sender.lock().unwrap().send(ServerInternalMessage::InferenceRequest((payload, tx)));
        let res = rx.recv().unwrap();
        match res {
            ServerInternalMessage::InferenceRequest(_) => panic!("Got inference request somehow"),
            ServerInternalMessage::InferenceResponse(vec) => {
                let http_response = SimpleServer::format_result_http(&vec);
                let _ = stream.write_all(http_response.as_bytes()).unwrap();
            }
        }
    }

    pub fn serve_forever_simple(&self) {
        println!("Start serving at {}:{}", self.addr, self.port);
        for stream in self.listener.incoming() {
            let stream = stream.unwrap();
            self.handle_connection(stream);
        }
    }

    pub fn serve_forever_threads(&self) {
        println!("Start serving at {}:{}", self.addr, self.port);
        let model = self.model.clone();
        let rx = self.rx.clone();
        let worker_thread_handle = thread::spawn(|| 
            {
                SimpleServer::inference_worker(model, rx)
            }
        );
        for stream in self.listener.incoming() {
            let tx = self.tx.clone();
            self.pool.execute(|| SimpleServer::connection_worker(stream.unwrap(), tx));
        }
    }
}
