// src/lib.rs
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ort)]
    type InferenceSession;

    #[wasm_bindgen(js_namespace = ort, js_name = "InferenceSession")]
    async fn create(model_path: &str) -> InferenceSession;

    #[wasm_bindgen(method)]
    async fn run(this: &InferenceSession, feeds: JsValue) -> JsValue;
}

#[wasm_bindgen]
pub struct Model {
    session: InferenceSession,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub async fn new(model_path: &str) -> Result<Model, JsValue> {
        let session = create(model_path).await;
        Ok(Model { session })
    }

    pub async fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>, JsValue> {
        // Create input tensor
        let input_tensor = {
            let obj = js_sys::Object::new();
            let array = js_sys::Float32Array::from(&input[..]);
            js_sys::Reflect::set(&obj, &"data".into(), &array)?;
            js_sys::Reflect::set(
                &obj,
                &"dims".into(),
                &js_sys::Array::of2(&1.into(), &4.into()),
            )?;
            obj
        };

        // Create feeds
        let feeds = js_sys::Object::new();
        js_sys::Reflect::set(&feeds, &"input".into(), &input_tensor)?;

        // Run inference
        let outputs = self.session.run(feeds.into()).await;

        // Extract output
        let output_data = js_sys::Reflect::get(&outputs, &"output".into())?;
        let array = js_sys::Float32Array::new(&output_data);
        let mut result = vec![0.0; array.length() as usize];
        array.copy_to(&mut result);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    use web_sys::console;

    // Configure to run in browser
    wasm_bindgen_test_configure!(run_in_browser);

    // Helper function to check if ONNX Runtime is loaded
    fn is_ort_loaded() -> bool {
        let window = web_sys::window().unwrap();
        js_sys::Reflect::has(&window, &"ort".into()).unwrap_or(false)
    }

    #[wasm_bindgen_test]
    async fn test_iris_model() {
        // Verify ONNX Runtime is available
        assert!(is_ort_loaded(), "ONNX Runtime (ort) not loaded! Make sure you include onnxruntime-web in your test HTML");

        // Test with known Iris setosa sample
        let setosa_input = vec![5.1, 3.5, 1.4, 0.2];

        // Load and run model
        match Model::new("../models/iris/iris_model.onnx").await {
            Ok(model) => {
                console::log_1(&"Model loaded successfully".into());

                match model.predict(setosa_input.clone()).await {
                    Ok(output) => {
                        console::log_1(&format!("Input: {:?}", setosa_input).into());
                        console::log_1(&format!("Output: {:?}", output).into());

                        // Should predict class 0 (Setosa)
                        let predicted_class = output
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                            .unwrap();

                        assert_eq!(predicted_class, 0, "Should predict Setosa (class 0)");
                    }
                    Err(e) => panic!("Prediction failed: {:?}", e),
                }
            }
            Err(e) => panic!("Model loading failed: {:?}", e),
        }
    }
}
