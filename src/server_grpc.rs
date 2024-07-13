use std::sync::Arc;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::nn::LinearLayer;
use ll_server::ll_service_server::{LlService, LlServiceServer};
use ll_server::{LlRequest, LlResponse};

pub mod ll_server {
    tonic::include_proto!("ll_server");
}

#[tonic::async_trait]
impl LlService for GrpcServer {
    async fn ll_dot(&self, request: Request<LlRequest>) -> Result<Response<LlResponse>, Status> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ServerInternalMessage>(100);
        let payload = request.into_inner().x;
        if let Some(ref rtx) = self.tx {
            rtx.send(ServerInternalMessage::InferenceRequest(payload, tx))
                .await
                .unwrap();
        };
        let res = rx.recv().await.unwrap();
        match res {
            ServerInternalMessage::InferenceResponse(vec) => {
                return Ok(Response::new(LlResponse { output: vec }));
            }
            _ => panic!("Error receiving message in connection worker"),
        }
    }
}

enum ServerInternalMessage {
    InferenceResponse(Vec<f32>),
    InferenceRequest(Vec<f32>, tokio::sync::mpsc::Sender<ServerInternalMessage>),
}

/// 🚀 blazingly fast
pub struct GrpcServer {
    addr: String,
    port: String,
    model: Arc<LinearLayer>,
    tx: Option<Arc<tokio::sync::mpsc::Sender<ServerInternalMessage>>>,
}

impl GrpcServer {
    pub fn new(addr: &str, port: &str, model: LinearLayer) -> Self {
        return Self {
            addr: addr.to_owned(),
            port: port.to_string(),
            model: Arc::new(model),
            tx: None,
        };
    }

    pub async fn serve(mut self: Self, batch_size: Option<usize>) {
        let (tx, rx) = tokio::sync::mpsc::channel::<ServerInternalMessage>(100);
        let tx = Arc::new(tx);
        let model = self.model.clone();
        let _worker_thread_handle =
            tokio::spawn( async move { GrpcServer::inference_worker(model, rx, batch_size).await });
        self.tx = Some(tx);
        let full_addr = format!("{}:{}", self.addr, self.port);
        Server::builder()
            .add_service(LlServiceServer::new(self))
            .serve(full_addr.parse().unwrap())
            .await
            .unwrap()
    }

    async fn inference_worker(
        model: Arc<LinearLayer>,
        mut inference_worker_receiver: tokio::sync::mpsc::Receiver<ServerInternalMessage>,
        batch_size: Option<usize>
    ) {
        loop {
            let mut batch: Vec<Vec<f32>> = vec![];
            let mut send_channels: Vec<tokio::sync::mpsc::Sender<ServerInternalMessage>> = vec![];
            let batch_size = batch_size.unwrap_or(8);
            let mut batch_recv = Vec::with_capacity(batch_size);

            inference_worker_receiver
                .recv_many(&mut batch_recv, batch_size)
                .await;

            for elem in batch_recv {
                match elem {
                    ServerInternalMessage::InferenceRequest(x, tx) => {
                        batch.push(x);
                        send_channels.push(tx);
                    }
                    _ => panic!("Penis"),
                }
            }

            let results: Vec<Vec<f32>> = model.forward(&batch);
            for (result, channel) in results.into_iter().zip(send_channels.iter()) {
                channel
                    .send(ServerInternalMessage::InferenceResponse(result))
                    .await
                    .unwrap();
            }
        }
    }
}
