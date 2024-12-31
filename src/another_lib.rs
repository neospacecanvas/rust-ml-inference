use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

// Enable console error logging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Simple function to read ONNX file bytes
#[wasm_bindgen]
pub async fn read_onnx_bytes(file_path: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(file_path)).await?;

    let response: web_sys::Response = response.dyn_into()?;
    let array_buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes = uint8_array.to_vec();

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    use web_sys::console;
    // C
    // onfigure to run in browser
    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn simple_test() {
        assert_eq!(2 + 2, 4);
    }

    #[wasm_bindgen]
    extern "C" {
        type InferenceSession;

        #[wasm_bindgen(constructor)]
        fn new() -> InferenceSession;

        #[wasm_bindgen(method, catch)]
        async fn loadModel(this: &InferenceSession, buffer: Vec<u8>) -> Result<(), JsValue>;
    }

    #[wasm_bindgen_test]
    async fn test_load_model() {
        let window = web_sys::window().unwrap();
        let response = wasm_bindgen_futures::JsFuture::from(
            window.fetch_with_str("../models/iris/iris_model.onnx"),
        )
        .await
        .unwrap();

        let response: web_sys::Response = response.dyn_into().unwrap();
        let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().unwrap())
            .await
            .unwrap();
        let uint8_array = js_sys::Uint8Array::new(&buffer);
        let bytes = uint8_array.to_vec();

        let session = InferenceSession::new();
        let result = session.loadModel(bytes).await;
        assert!(result.is_ok(), "Should load ONNX model successfully");
    }

    #[wasm_bindgen_test]
    async fn test_onnx_file() {
       let window = web_sys::window().unwrap();
       let response = wasm_bindgen_futures::JsFuture::from(
           window.fetch_with_str("../models/iris/iris_model.onnx"),
       )
       .await
       .unwrap();
    
       let response: web_sys::Response = response.dyn_into().unwrap();
       let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().unwrap())
           .await
           .unwrap();
       let uint8_array = js_sys::Uint8Array::new(&buffer);
       let bytes = uint8_array.to_vec();
    
       // Log first few bytes to see what we're getting
       log(&format!("First 10 bytes: {:?}", &bytes[..10]));
       // Log total size
       log(&format!("Total file size: {} bytes", bytes.len()));
    }

    #[wasm_bindgen_test]
    async fn test_read_onnx() {
       match read_onnx_bytes("../models/iris/iris_model.onnx").await {
           Ok(bytes) => {
               // Check if bytes look like an ONNX file (magic number)
               if bytes.len() >= 4 && &bytes[0..4] == b"ONNX" {
                   log("Successfully read ONNX file!");
                   log(&format!("File size: {} bytes", bytes.len()));
               } else {
                   log("File read but doesn't appear to be ONNX format");
               }
           }
           Err(e) => log(&format!("Error reading file: {:?}", e)),
       }
    }
}
