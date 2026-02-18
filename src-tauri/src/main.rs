#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend_supervisor;

use backend_supervisor::{spawn_backend, status, stop_backend, SupervisorState};
use std::sync::{Arc, Mutex};
use tauri::{CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu};

struct AppState {
  supervisor: Arc<Mutex<SupervisorState>>,
  base_url: Arc<Mutex<Option<String>>>,
}

#[tauri::command]
fn get_backend_base_url(state: tauri::State<AppState>) -> Result<String, String> {
  state
    .base_url
    .lock()
    .map_err(|_| "lock poisoned".to_string())?
    .clone()
    .ok_or_else(|| "backend not initialized".to_string())
}

#[tauri::command]
fn get_backend_status(state: tauri::State<AppState>) -> backend_supervisor::BackendStatus {
  status(state.supervisor.clone())
}

#[tauri::command]
fn get_backend_logs(_state: tauri::State<AppState>) -> Vec<String> {
  Vec::new()
}

#[tauri::command]
fn restart_backend(state: tauri::State<AppState>) -> Result<String, String> {
  stop_backend(state.supervisor.clone());
  let url = spawn_backend(state.supervisor.clone())?;
  *state.base_url.lock().map_err(|_| "lock poisoned".to_string())? = Some(url.clone());
  Ok(url)
}

fn main() {
  let supervisor = Arc::new(Mutex::new(SupervisorState::default()));
  let initial_url = spawn_backend(supervisor.clone()).ok();
  let base_url = Arc::new(Mutex::new(initial_url));

  let tray_menu = SystemTrayMenu::new()
    .add_item(CustomMenuItem::new("show", "Show"))
    .add_item(CustomMenuItem::new("restart", "Restart backend"))
    .add_item(CustomMenuItem::new("quit", "Quit"));

  tauri::Builder::default()
    .manage(AppState { supervisor: supervisor.clone(), base_url: base_url.clone() })
    .invoke_handler(tauri::generate_handler![get_backend_base_url, get_backend_status, get_backend_logs, restart_backend])
    .system_tray(SystemTray::new().with_menu(tray_menu))
    .on_system_tray_event(|app, event| match event {
      SystemTrayEvent::MenuItemClick { id, .. } if id.as_str() == "show" => {
        if let Some(w) = app.get_window("main") { let _ = w.show(); let _ = w.set_focus(); }
      }
      SystemTrayEvent::MenuItemClick { id, .. } if id.as_str() == "restart" => {
        let state: tauri::State<AppState> = app.state();
        let _ = restart_backend(state);
      }
      SystemTrayEvent::MenuItemClick { id, .. } if id.as_str() == "quit" => app.exit(0),
      _ => {}
    })
    .on_window_event(move |event| {
      if let tauri::WindowEvent::CloseRequested { .. } = event.event() {
        stop_backend(supervisor.clone());
      }
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
