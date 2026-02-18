use serde::Serialize;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

#[derive(Default)]
pub struct SupervisorState {
  pub child: Option<Child>,
  pub port: Option<u16>,
  pub last_health_ok_at: Option<u64>,
  pub logs: Vec<String>,
}

#[derive(Serialize)]
pub struct BackendStatus {
  pub state: String,
  pub pid: Option<u32>,
  pub port: Option<u16>,
  pub last_health_ok_at: Option<u64>,
}

fn choose_free_port() -> u16 {
  std::net::TcpListener::bind("127.0.0.1:0")
    .expect("bind free port")
    .local_addr()
    .expect("local addr")
    .port()
}

pub fn spawn_backend(state: Arc<Mutex<SupervisorState>>) -> Result<String, String> {
  let port = choose_free_port();
  let mut cmd = Command::new("backend_dist/orchestrator.exe");
  cmd
    .arg("--host")
    .arg("127.0.0.1")
    .arg("--port")
    .arg(port.to_string())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

  let child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
  {
    let mut guard = state.lock().map_err(|_| "lock poisoned".to_string())?;
    guard.port = Some(port);
    guard.child = Some(child);
  }

  let base_url = format!("http://127.0.0.1:{port}");
  for _ in 0..30 {
    if let Ok(resp) = reqwest::blocking::get(format!("{base_url}/healthz")) {
      if resp.status().is_success() {
        let mut guard = state.lock().map_err(|_| "lock poisoned".to_string())?;
        guard.last_health_ok_at = Some(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs());
        return Ok(base_url);
      }
    }
    std::thread::sleep(Duration::from_millis(500));
  }
  Err("backend health probe timed out".to_string())
}

pub fn stop_backend(state: Arc<Mutex<SupervisorState>>) {
  if let Ok(mut guard) = state.lock() {
    if let Some(mut child) = guard.child.take() {
      let _ = child.kill();
    }
  }
}

pub fn status(state: Arc<Mutex<SupervisorState>>) -> BackendStatus {
  if let Ok(guard) = state.lock() {
    BackendStatus {
      state: if guard.child.is_some() { "running".to_string() } else { "stopped".to_string() },
      pid: guard.child.as_ref().map(|c| c.id()),
      port: guard.port,
      last_health_ok_at: guard.last_health_ok_at,
    }
  } else {
    BackendStatus { state: "error".to_string(), pid: None, port: None, last_health_ok_at: None }
  }
}
