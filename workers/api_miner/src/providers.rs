use crate::{now_ms, Assignment, AssignmentInput, Error, Result, MAX_MEDIA_BYTES};
use base64::{
    engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD as B64URL},
    Engine,
};
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;
use std::{io::Read, sync::Mutex};

pub struct ProviderSubmission {
    pub reference: String,
    pub response: Vec<u8>,
}
pub trait Provider {
    fn submit(&self, assignment: &Assignment) -> Result<ProviderSubmission>;
    fn collect(&self, assignment: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>>;
}

/// Direct adapters expose no caller-controlled base URL.  Contracts must map a
/// model to one approved provider endpoint before a job can be assigned.
pub struct DirectProvider {
    configured_provider: String,
    client: reqwest::blocking::Client,
    key: zeroize::Zeroizing<String>,
    kling_access_key: Option<zeroize::Zeroizing<String>>,
    kling_secret_key: Option<zeroize::Zeroizing<String>>,
    // The signed token is sensitive too. Keep it only inside the process and
    // refresh before expiry; it is never persisted with assignment state.
    kling_token: Mutex<Option<(i64, zeroize::Zeroizing<String>)>>,
    google_project: Option<String>,
    google_location: Option<String>,
    google_output_bucket: Option<String>,
    #[cfg(test)]
    test_origin: Option<reqwest::Url>,
}
impl DirectProvider {
    /// Queue status/result routes are addressed to a fal app, never a nested
    /// model endpoint. Keep the app mapping explicit so adding a model needs
    /// review rather than a lossy path heuristic.
    pub(crate) fn fal_app(model: &str) -> Result<&'static str> {
        match model {
            "bytedance/seedance-2.5/text-to-video" => Ok("bytedance/seedance-2.5"),
            "bytedance/seedance-2.0/text-to-video"
            | "bytedance/seedance-2.0/image-to-video"
            | "bytedance/seedance-2.0/fast/text-to-video"
            | "bytedance/seedance-2.0/fast/image-to-video" => Ok("bytedance/seedance-2.0"),
            "fal-ai/kling-video/v3/turbo/standard/text-to-video" => Ok("fal-ai/kling-video"),
            "fal-ai/veo3.1"
            | "fal-ai/veo3.1/fast"
            | "fal-ai/veo3.1/image-to-video"
            | "fal-ai/veo3.1/fast/image-to-video" => Ok("fal-ai/veo3.1"),
            "minimax/h3-max/text-to-video" => Ok("minimax/h3-max"),
            "alibaba/wan-3.0/text-to-video" => Ok("alibaba/wan-3.0"),
            "blackforestlabs/flux-3-max/text-to-video"
            | "blackforestlabs/flux-3-max/image-to-video" => Ok("blackforestlabs/flux-3-max"),
            "fal-ai/nano-banana-pro" | "fal-ai/nano-banana-pro/edit" => {
                Ok("fal-ai/nano-banana-pro")
            }
            "google/nano-banana-2.1" | "google/nano-banana-2.1/edit" => {
                Ok("google/nano-banana-2.1")
            }
            "fal-ai/flux-2/flash" | "fal-ai/flux-2/flash/edit" => Ok("fal-ai/flux-2"),
            "bytedance/seedream/v5/lite/text-to-image" | "bytedance/seedream/v5/lite/edit" => {
                Ok("bytedance/seedream")
            }
            _ => Err(Error::Protocol("provider_model_not_approved")),
        }
    }
    pub(crate) fn fal_status_url(model: &str, reference: &str) -> Result<reqwest::Url> {
        reqwest::Url::parse(&format!(
            "https://queue.fal.run/{}/requests/{reference}/status",
            Self::fal_app(model)?
        ))
        .map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    pub(crate) fn fal_result_url(model: &str, reference: &str) -> Result<reqwest::Url> {
        reqwest::Url::parse(&format!(
            "https://queue.fal.run/{}/requests/{reference}",
            Self::fal_app(model)?
        ))
        .map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    pub(crate) fn approved_model(provider: &str, model: &str) -> bool {
        match provider {
            "byteplus" => matches!(
                model,
                "dreamina-seedance-2-5-260628"
                    | "dreamina-seedance-2-0-260128"
                    | "dreamina-seedance-2-0-mini-260615"
                    | "dreamina-seedance-2-0-fast-260128"
            ),
            "kling" => model == "kling-v2-6",
            "google-cloud" => matches!(model, "veo-3.1-generate-001" | "veo-3.1-fast-generate-001"),
            "atlascloud" => matches!(
                model,
                "kwaivgi/kling-v4.0/text-to-video"
                    | "kwaivgi/kling-v4.0/image-to-video"
                    | "kwaivgi/kling-image-v4.0/text-to-image"
                    | "minimax/h3-developer/text-to-video"
            ),
            // Every fal route is pinned explicitly in fal_app; adding a model
            // is a reviewed release change, never a caller-selected endpoint.
            "fal" => Self::fal_app(model).is_ok(),
            _ => false,
        }
    }
    pub fn from_env(provider: &str) -> Result<Self> {
        let variable = match provider {
            "byteplus" => "BYTEPLUS_API_KEY",
            "kling" => "",
            "google-cloud" => "GOOGLE_CLOUD_ACCESS_TOKEN",
            "fal" => "FAL_KEY",
            "atlascloud" => "ATLASCLOUD_API_KEY",
            _ => return Err(Error::Protocol("unsupported_provider")),
        };
        let key = if provider == "kling" {
            String::new()
        } else {
            std::env::var(variable).map_err(|_| Error::Protocol("provider_key_unavailable"))?
        };
        if provider != "kling" && key.trim().is_empty() {
            return Err(Error::Protocol("provider_key_unavailable"));
        }
        let kling_access_key = if provider == "kling" {
            Some(zeroize::Zeroizing::new(
                std::env::var("KLING_ACCESS_KEY")
                    .map_err(|_| Error::Protocol("kling_access_key_unavailable"))?,
            ))
        } else {
            None
        };
        let kling_secret_key = if provider == "kling" {
            Some(zeroize::Zeroizing::new(
                std::env::var("KLING_SECRET_KEY")
                    .map_err(|_| Error::Protocol("kling_secret_key_unavailable"))?,
            ))
        } else {
            None
        };
        if kling_access_key
            .as_deref()
            .is_some_and(|key| key.trim().is_empty())
            || kling_secret_key
                .as_deref()
                .is_some_and(|key| key.trim().is_empty())
        {
            return Err(Error::Protocol("kling_credentials_unavailable"));
        }
        let project = std::env::var("GOOGLE_CLOUD_PROJECT").ok();
        let location = std::env::var("GOOGLE_CLOUD_LOCATION").ok();
        let bucket = std::env::var("GOOGLE_CLOUD_OUTPUT_BUCKET").ok();
        let safe = |v: &str| {
            v.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
                && v.len() >= 3
        };
        if provider == "google-cloud"
            && (!project.as_deref().is_some_and(safe)
                || !location.as_deref().is_some_and(safe)
                || !bucket.as_deref().is_some_and(safe))
        {
            return Err(Error::Protocol("google_cloud_project_location_or_bucket"));
        }
        Ok(Self {
            configured_provider: provider.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            key: zeroize::Zeroizing::new(key),
            kling_access_key,
            kling_secret_key,
            kling_token: Mutex::new(None),
            google_project: project,
            google_location: location,
            google_output_bucket: bucket,
            #[cfg(test)]
            test_origin: None,
        })
    }
    #[cfg(test)]
    pub(crate) fn fixture(origin: &str, configured_provider: &str) -> Self {
        Self {
            configured_provider: configured_provider.into(),
            client: reqwest::blocking::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            key: zeroize::Zeroizing::new("fixture-key".into()),
            kling_access_key: Some(zeroize::Zeroizing::new("fixture-access".into())),
            kling_secret_key: Some(zeroize::Zeroizing::new("fixture-secret".into())),
            kling_token: Mutex::new(None),
            google_project: Some("fixture-project".into()),
            google_location: Some("us-central1".into()),
            google_output_bucket: Some("fixture-bucket".into()),
            test_origin: Some(reqwest::Url::parse(origin).unwrap()),
        }
    }
    fn authorization(&self, provider: &str) -> Result<String> {
        if provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        if provider == "fal" {
            return Ok(format!("Key {}", self.key.as_str()));
        }
        if provider != "kling" {
            return Ok(format!("Bearer {}", self.key.as_str()));
        }
        // Kling v2.6 uses the legacy API design: generate a short-lived HS256
        // JWT from the sealed AccessKey/SecretKey on every upstream call.
        let access = self
            .kling_access_key
            .as_deref()
            .ok_or(Error::Protocol("kling_access_key_unavailable"))?;
        let secret = self
            .kling_secret_key
            .as_deref()
            .ok_or(Error::Protocol("kling_secret_key_unavailable"))?;
        let now = now_ms() / 1_000;
        let mut cached = self
            .kling_token
            .lock()
            .map_err(|_| Error::Protocol("kling_token_lock"))?;
        if let Some((expires_at, token)) = cached.as_ref() {
            if *expires_at > now + 60 {
                return Ok(format!("Bearer {}", token.as_str()));
            }
        }
        let header = B64URL.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = B64URL.encode(serde_json::to_vec(
            &serde_json::json!({"iss":access,"exp":now + 1_800,"nbf":now - 5}),
        )?);
        let signed = format!("{header}.{payload}");
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(secret.as_bytes())
            .map_err(|_| Error::Protocol("kling_secret_key_invalid"))?;
        mac.update(signed.as_bytes());
        let token = zeroize::Zeroizing::new(format!(
            "{signed}.{}",
            B64URL.encode(mac.finalize().into_bytes())
        ));
        *cached = Some((now + 1_800, token));
        Ok(format!(
            "Bearer {}",
            cached.as_ref().expect("just set").1.as_str()
        ))
    }
    fn endpoint(&self, provider: &str, model: &str, modality: &str) -> Result<reqwest::Url> {
        if !Self::approved_model(provider, model) {
            return Err(Error::Protocol("provider_model_not_approved"));
        }
        #[cfg(test)]
        if let Some(origin) = &self.test_origin {
            return origin
                .join(&format!("upstream/{provider}/submit"))
                .map_err(|_| Error::Protocol("provider_endpoint"));
        }
        let endpoint=match provider {
   "byteplus"=>"https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks",
   "kling"=>if modality=="image-to-video"{"https://api-singapore.klingai.com/v1/videos/image2video"}else{"https://api-singapore.klingai.com/v1/videos/text2video"},
   "google-cloud"=>return reqwest::Url::parse(&format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predictLongRunning",self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_project.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,model)).map_err(|_|Error::Protocol("provider_endpoint")),
   "atlascloud"=>if matches!(modality,"text-to-image"|"image-to-image"){"https://api.atlascloud.ai/api/v1/model/generateImage"}else{"https://api.atlascloud.ai/api/v1/model/generateVideo"},
   "fal"=>{
       if model.contains("..") || !model.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b==b'/' || b==b'.' || b==b'_' || b==b'-') { return Err(Error::Protocol("provider_model_invalid")); }
       return reqwest::Url::parse(&format!("https://queue.fal.run/{model}")).map_err(|_|Error::Protocol("provider_endpoint"));
   }
   _=>return Err(Error::Protocol("unsupported_provider")),
  };
        reqwest::Url::parse(endpoint).map_err(|_| Error::Protocol("provider_endpoint"))
    }
    fn reference(value: &Value) -> Option<String> {
        ["/id", "/task_id", "/data/id", "/data/task_id", "/name", "/request_id"]
            .iter()
            .find_map(|path| {
                value
                    .pointer(path)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
    }
    fn status_url(&self, a: &Assignment, reference: &str) -> Result<reqwest::Url> {
        let encoded: String = reference
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/') {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect();
        #[cfg(test)]
        if let Some(origin) = &self.test_origin {
            return origin
                .join(&format!("upstream/{}/status/{encoded}", a.provider))
                .map_err(|_| Error::Protocol("provider_reference_invalid"));
        }
        let endpoint=match a.provider.as_str(){
     "byteplus"=>format!("https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{encoded}"),
     "kling"=>format!("https://api-singapore.klingai.com/v1/videos/{}/{encoded}",if a.modality=="image-to-video"{"image2video"}else{"text2video"}),
     // Vertex Veo is polled through the documented model operation method,
     // rather than treating an operation name as a generic REST resource.
     "google-cloud"=>format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:fetchPredictOperation",self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_project.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,a.model),
     "atlascloud"=>format!("https://api.atlascloud.ai/api/v1/model/prediction/{encoded}"),
     "fal"=>return Self::fal_status_url(&a.model,&encoded),
     _=>return Err(Error::Protocol("unsupported_provider")),
   };
        reqwest::Url::parse(&endpoint).map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    pub(crate) fn payload(&self, a: &Assignment) -> Result<Value> {
        let object = a
            .request
            .as_object()
            .ok_or(Error::Protocol("provider_request_invalid"))?;
        let is_image_output = matches!(a.modality.as_str(), "text-to-image" | "image-to-image");
        let needs_image_input = matches!(a.modality.as_str(), "image-to-video" | "image-to-image");
        if object.keys().any(|key| {
            let allowed = if is_image_output {
                matches!(
                    key.as_str(),
                    "prompt" | "aspectRatio" | "resolution" | "imageSize"
                ) || (key == "imageId" && needs_image_input)
            } else {
                matches!(key.as_str(), "prompt" | "duration" | "aspectRatio")
                    || (key == "imageId" && needs_image_input)
            };
            !allowed
        }) {
            return Err(Error::Protocol("provider_request_unsupported"));
        }
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|v| !v.trim().is_empty() && v.len() <= 8_000)
            .ok_or(Error::Protocol("provider_prompt_invalid"))?
            .trim();
        let image = || -> Result<&AssignmentInput> {
            a.inputs
                .first()
                .ok_or(Error::Protocol("assignment_image_required"))
        };
        if needs_image_input {
            if let Some(image_id) = object.get("imageId") {
                if image_id.as_str() != a.inputs.first().map(|input| input.id.as_str()) {
                    return Err(Error::Protocol("assignment_image_id_mismatch"));
                }
            }
        }
        let data_uri =
            |input: &AssignmentInput| format!("data:{};base64,{}", input.mime, input.bytes_base64);

        if is_image_output {
            let img_aspect = object.get("aspectRatio").and_then(Value::as_str).filter(|v| {
                matches!(
                    *v,
                    "auto"
                        | "21:9"
                        | "16:9"
                        | "3:2"
                        | "4:3"
                        | "5:4"
                        | "1:1"
                        | "4:5"
                        | "3:4"
                        | "2:3"
                        | "9:16"
                )
            });
            let img_size = object.get("imageSize").and_then(Value::as_str).filter(|v| {
                matches!(
                    *v,
                    "square"
                        | "square_hd"
                        | "portrait_4_3"
                        | "portrait_16_9"
                        | "landscape_4_3"
                        | "landscape_16_9"
                )
            });
            let img_res = object.get("resolution").and_then(Value::as_str);
            return Ok(match (a.provider.as_str(), a.model.as_str(), a.modality.as_str()) {
                ("atlascloud", "kwaivgi/kling-image-v4.0/text-to-image", "text-to-image") => {
                    let resolution = match img_res.unwrap_or("2k") {
                        "2k" | "4k" => img_res.unwrap_or("2k"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    serde_json::json!({"model":a.model,"prompt":prompt,"resolution":resolution,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("auto")})
                }
                ("atlascloud", "kwaivgi/kling-image-v4.0/text-to-image", "image-to-image") => {
                    let resolution = match img_res.unwrap_or("2k") {
                        "2k" | "4k" => img_res.unwrap_or("2k"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    let input = image()?;
                    serde_json::json!({"model":a.model,"prompt":prompt,"resolution":resolution,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("auto"),"reference_images":[data_uri(input)]})
                }
                ("fal", "fal-ai/nano-banana-pro", "text-to-image") => {
                    let resolution = match img_res.unwrap_or("1K") {
                        "1K" | "2K" | "4K" => img_res.unwrap_or("1K"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    serde_json::json!({"prompt":prompt,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("16:9"),"resolution":resolution,"output_format":"png","safety_tolerance":"4","enable_web_search":false,"limit_generations":true})
                }
                ("fal", "fal-ai/nano-banana-pro/edit", "image-to-image") => {
                    let resolution = match img_res.unwrap_or("1K") {
                        "1K" | "2K" | "4K" => img_res.unwrap_or("1K"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    let input = image()?;
                    serde_json::json!({"prompt":prompt,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("auto"),"resolution":resolution,"output_format":"png","safety_tolerance":"4","enable_web_search":false,"limit_generations":true,"image_urls":[data_uri(input)]})
                }
                ("fal", "google/nano-banana-2.1", "text-to-image") => {
                    let resolution = match img_res.unwrap_or("1K") {
                        "1K" | "2K" | "4K" => img_res.unwrap_or("1K"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    serde_json::json!({"prompt":prompt,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("auto"),"resolution":resolution,"output_format":"png","safety_tolerance":"4","enable_web_search":false,"limit_generations":true,"thinking_level":"medium"})
                }
                ("fal", "google/nano-banana-2.1/edit", "image-to-image") => {
                    let resolution = match img_res.unwrap_or("1K") {
                        "1K" | "2K" | "4K" => img_res.unwrap_or("1K"),
                        _ => return Err(Error::Protocol("provider_request_unsupported")),
                    };
                    let input = image()?;
                    serde_json::json!({"prompt":prompt,"num_images":1,"aspect_ratio":img_aspect.unwrap_or("auto"),"resolution":resolution,"output_format":"png","safety_tolerance":"4","enable_web_search":false,"limit_generations":true,"thinking_level":"medium","image_urls":[data_uri(input)]})
                }
                ("fal", "fal-ai/flux-2/flash", "text-to-image") => {
                    serde_json::json!({"prompt":prompt,"guidance_scale":2.5,"image_size":img_size.unwrap_or("landscape_4_3"),"num_images":1,"enable_prompt_expansion":false,"enable_safety_checker":true,"output_format":"png"})
                }
                ("fal", "fal-ai/flux-2/flash/edit", "image-to-image") => {
                    let input = image()?;
                    serde_json::json!({"prompt":prompt,"guidance_scale":2.5,"image_size":img_size.unwrap_or("square_hd"),"num_images":1,"enable_prompt_expansion":false,"enable_safety_checker":true,"output_format":"png","image_urls":[data_uri(input)]})
                }
                ("fal", "bytedance/seedream/v5/lite/text-to-image", "text-to-image") => {
                    serde_json::json!({"prompt":prompt,"image_size":"auto_2K","num_images":1,"output_format":"png","enable_safety_checker":true})
                }
                ("fal", "bytedance/seedream/v5/lite/edit", "image-to-image") => {
                    let input = image()?;
                    serde_json::json!({"prompt":prompt,"image_size":"auto_2K","num_images":1,"output_format":"png","enable_safety_checker":true,"image_urls":[data_uri(input)]})
                }
                _ => return Err(Error::Protocol("provider_model_not_approved")),
            });
        }

        let duration = object
            .get("duration")
            .and_then(Value::as_u64)
            .filter(|v| *v >= 1 && *v <= 15)
            .ok_or(Error::Protocol("provider_duration_invalid"))?;
        let aspect = object
            .get("aspectRatio")
            .and_then(Value::as_str)
            .filter(|v| matches!(*v, "16:9" | "9:16"));
        let wide_aspect = object
            .get("aspectRatio")
            .and_then(Value::as_str)
            .filter(|v| matches!(*v, "16:9" | "9:16" | "1:1" | "auto"));
        Ok(match a.provider.as_str() {
            // ModelArk's video endpoint names the aspect field `ratio`; this is a
            // deliberately text-only v1 contract (no caller URLs or arbitrary JSON).
            "byteplus" if a.modality == "image-to-video" => {
                let input = image()?;
                // Seedance 2.5 derives the output ratio from its first frame.
                // ModelArk rejects a fixed ratio together with first_frame.
                serde_json::json!({"model":a.model,"content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":data_uri(input)},"role":"first_frame"}],"duration":duration,"ratio":"adaptive","resolution":"720p","watermark":false})
            }
            "byteplus" => {
                serde_json::json!({"model":a.model,"content":[{"type":"text","text":prompt}],"duration":duration,"ratio":aspect,"resolution":"720p","watermark":false})
            }
            "kling" if a.modality == "image-to-video" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"model_name":a.model,"image":input.bytes_base64,"prompt":prompt,"duration":duration.to_string(),"mode":"pro","sound":"off"})
            }
            "kling" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"model_name":a.model,"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect})
            }
            "google-cloud" => {
                let mut instance = serde_json::json!({"prompt":prompt});
                if a.modality == "image-to-video" {
                    let input = a
                        .inputs
                        .first()
                        .ok_or(Error::Protocol("assignment_image_required"))?;
                    // Vertex's image-to-video operation consumes the source
                    // frame as `instance.image`; referenceImages are a
                    // different asset/style-conditioning capability.
                    instance["image"] = serde_json::json!({"bytesBase64Encoded":input.bytes_base64,"mimeType":input.mime});
                }
                serde_json::json!({"instances":[instance],"parameters":{"aspectRatio":aspect,"durationSeconds":duration,"storageUri":format!("gs://{}/attelier-api-miner/",self.google_output_bucket.as_deref().ok_or(Error::Protocol("google_cloud_project_location_or_bucket"))?),"sampleCount":1}})
            }
            "atlascloud" if a.model == "kwaivgi/kling-v4.0/text-to-video" && a.modality == "text-to-video" => {
                if !(3..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"model":a.model,"prompt":prompt,"resolution":"1080p","duration":duration,"dynamic_range":"sdr","aspect_ratio":wide_aspect.filter(|v| *v != "auto").unwrap_or("16:9")})
            }
            "atlascloud" if a.model == "kwaivgi/kling-v4.0/image-to-video" && a.modality == "image-to-video" => {
                if !(3..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"model":a.model,"prompt":prompt,"resolution":"1080p","duration":duration,"dynamic_range":"sdr","image":data_uri(input)})
            }
            "atlascloud" if a.model == "minimax/h3-developer/text-to-video" && a.modality == "text-to-video" => {
                if !(5..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"model":a.model,"prompt":prompt,"duration":duration,"resolution":"768P","ratio":wide_aspect.filter(|v| *v != "auto").unwrap_or("16:9"),"prompt_expansion":false})
            }
            "atlascloud" => return Err(Error::Protocol("provider_model_not_approved")),
            // Each fal route owns its own body schema. In particular, the
            // Seedance Queue OpenAPI uses string durations; do not infer a
            // payload shape from a model family or an older provider API.
            "fal" if a.model.starts_with("fal-ai/veo3.1") && a.modality == "image-to-video" => {
                if !a.model.ends_with("image-to-video") {
                    return Err(Error::Protocol("provider_model_not_approved"));
                }
                if !matches!(duration, 4 | 6 | 8) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"prompt":prompt,"image_url":data_uri(input),"duration":format!("{duration}s"),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model.starts_with("fal-ai/veo3.1") && a.modality == "text-to-video" => {
                if a.model.ends_with("image-to-video") {
                    return Err(Error::Protocol("provider_model_not_approved"));
                }
                if !matches!(duration, 4 | 6 | 8) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":format!("{duration}s"),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model.starts_with("fal-ai/kling-video/") && a.modality == "text-to-video" => {
                if !(3..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect})
            }
            "fal"
                if matches!(
                    a.model.as_str(),
                    "bytedance/seedance-2.0/image-to-video"
                        | "bytedance/seedance-2.0/fast/image-to-video"
                ) && a.modality == "image-to-video" =>
            {
                if !(4..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"prompt":prompt,"image_url":data_uri(input),"duration":duration.to_string(),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal"
                if matches!(
                    a.model.as_str(),
                    "bytedance/seedance-2.5/text-to-video"
                        | "bytedance/seedance-2.0/text-to-video"
                        | "bytedance/seedance-2.0/fast/text-to-video"
                ) && a.modality == "text-to-video" =>
            {
                // The provider supports up to 30 seconds on 2.5. The worker keeps
                // the existing 15-second assignment cap to bound spend.
                if !(4..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model == "minimax/h3-max/text-to-video" && a.modality == "text-to-video" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":wide_aspect.filter(|v| *v != "auto").unwrap_or("16:9"),"resolution":"768P","prompt_expansion_mode":"balanced"})
            }
            "fal" if a.model == "alibaba/wan-3.0/text-to-video" && a.modality == "text-to-video" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":wide_aspect.filter(|v| *v != "auto").unwrap_or("16:9"),"resolution":"720p"})
            }
            "fal"
                if a.model == "blackforestlabs/flux-3-max/text-to-video"
                    && a.modality == "text-to-video" =>
            {
                if !(5..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration,"aspect_ratio":wide_aspect.unwrap_or("16:9"),"resolution":"720p","generate_audio":true,"reasoning_level":"high"})
            }
            "fal"
                if a.model == "blackforestlabs/flux-3-max/image-to-video"
                    && a.modality == "image-to-video" =>
            {
                if !(5..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"prompt":prompt,"image_url":data_uri(input),"duration":duration,"aspect_ratio":wide_aspect.unwrap_or("auto"),"resolution":"720p","generate_audio":true,"reasoning_level":"high"})
            }
            "fal" => return Err(Error::Protocol("provider_model_not_approved")),
            _ => return Err(Error::Protocol("unsupported_provider")),
        })
    }
    pub(crate) fn output_url(&self, provider: &str, value: &Value) -> Option<String> {
        let candidates = [
            "/content/video_url",
            "/video/url",
            "/images/0/url",
            "/image/url",
            "/data/task_result/videos/0/url",
            "/data/outputs/0",
            "/response/generatedVideos/0/video/uri",
            "/response/videos/0/gcsUri",
            "/output/url",
            "/response/video/url",
        ];
        let url = candidates
            .iter()
            .find_map(|p| value.pointer(p).and_then(Value::as_str))?;
        let parsed = reqwest::Url::parse(url).ok()?;
        #[cfg(test)]
        if self
            .test_origin
            .as_ref()
            .is_some_and(|origin| origin.origin() == parsed.origin())
        {
            return Some(url.to_owned());
        }
        let host = parsed.host_str()?;
        let child_or_root = |suffix: &str| {
            host == suffix
                || host
                    .strip_suffix(suffix)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        };
        let permitted = match provider {
            "byteplus" => {
                child_or_root("bytepluscdn.com")
                    || child_or_root("bytepluses.com")
                    || child_or_root("volces.com")
            }
            "kling" => child_or_root("klingai.com") || child_or_root("kuaishou.com"),
            "google-cloud" => {
                child_or_root("googleusercontent.com") || child_or_root("googleapis.com")
            }
            "atlascloud" => {
                child_or_root("atlascloud.ai") || host == "atlas-media.oss-us-west-1.aliyuncs.com"
            }
            // fal's own schemas show result files at both fal.media and its
            // dedicated Google storage bucket. Do not permit arbitrary GCS
            // hosts: this exact host is the documented result origin.
            "fal" => {
                child_or_root("fal.media")
                    || child_or_root("fal.ai")
                    || host == "storage.googleapis.com"
            }
            _ => false,
        };
        if parsed.scheme() == "https" && permitted {
            Some(url.to_owned())
        } else {
            None
        }
    }
    fn media(response: reqwest::blocking::Response) -> Result<Vec<u8>> {
        if response
            .content_length()
            .is_some_and(|n| n > MAX_MEDIA_BYTES as u64)
        {
            return Err(Error::Protocol("invalid_output_bytes"));
        }
        let mut media = Vec::new();
        response
            .take(MAX_MEDIA_BYTES as u64 + 1)
            .read_to_end(&mut media)?;
        if media.is_empty() || media.len() > MAX_MEDIA_BYTES {
            return Err(Error::Protocol("invalid_output_bytes"));
        }
        Ok(media)
    }
}
impl Provider for DirectProvider {
    fn submit(&self, a: &Assignment) -> Result<ProviderSubmission> {
        if a.provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        let endpoint = self.endpoint(&a.provider, &a.model, &a.modality)?;
        let authorization = self.authorization(&a.provider)?;
        let response = self
            .client
            .post(endpoint)
            .header("authorization", authorization)
            .header("idempotency-key", format!("{}:{}", a.job_id, a.attempt_id))
            .json(&self.payload(a)?)
            .send()?;
        let status = response.status();
        let raw = response.bytes()?.to_vec();
        if !status.is_success() {
            if matches!(status.as_u16(), 400 | 401 | 403 | 404 | 422) {
                let provider_code = serde_json::from_slice::<Value>(&raw)
                    .ok()
                    .and_then(|value| {
                        value
                            .pointer("/error/code")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .filter(|code| {
                        !code.is_empty()
                            && code.len() <= 80
                            && code.as_bytes()[0].is_ascii_alphabetic()
                            && code.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                            })
                    });
                return Err(Error::ProviderTerminal {
                    reason: "provider_rejected",
                    http_status: Some(status.as_u16()),
                    provider_code,
                });
            }
            return Err(Error::Provider(format!(
                "{}:{}",
                a.provider,
                status.as_u16()
            )));
        }
        let value: Value = serde_json::from_slice(&raw)?;
        let reference =
            Self::reference(&value).ok_or(Error::Protocol("provider_reference_missing"))?;
        Ok(ProviderSubmission {
            reference,
            response: raw,
        })
    }
    fn collect(&self, a: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>> {
        if a.provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis((a.expires_at - now_ms()).max(0) as u64);
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(Error::Protocol("provider_deadline"));
            }
            let authorization = self.authorization(&a.provider)?;
            let response = if a.provider == "google-cloud" {
                self.client
                    .post(self.status_url(a, &submission.reference)?)
                    .header("authorization", self.authorization("google-cloud")?)
                    .json(&serde_json::json!({"operationName":submission.reference}))
                    .send()?
            } else {
                self.client
                    .get(self.status_url(a, &submission.reference)?)
                    .header("authorization", authorization)
                    .send()?
            };
            let status = response.status();
            let raw = response.bytes()?.to_vec();
            if !status.is_success() {
                return Err(Error::Provider(format!(
                    "{}-poll:{}",
                    a.provider,
                    status.as_u16()
                )));
            }
            let value: Value = serde_json::from_slice(&raw)?;
            let status = if a.provider == "google-cloud" {
                if value.get("done").and_then(Value::as_bool) == Some(true) {
                    "done"
                } else {
                    "running"
                }
            } else {
                value
                    .pointer("/status")
                    .or_else(|| value.pointer("/data/status"))
                    .or_else(|| value.pointer("/data/task_status"))
                    .and_then(Value::as_str)
                    .unwrap_or("running")
            };
            let normalized_status = status.to_ascii_lowercase();
            if matches!(normalized_status.as_str(), "cancelled" | "canceled") {
                return Err(Error::ProviderTerminal {
                    reason: "provider_cancelled",
                    http_status: None,
                    provider_code: None,
                });
            }
            if matches!(normalized_status.as_str(), "failed" | "error")
                || (a.provider == "fal" && status == "COMPLETED" && value.get("error").is_some())
            {
                return Err(Error::ProviderTerminal {
                    reason: "provider_failed",
                    http_status: None,
                    provider_code: None,
                });
            }
            if matches!(
                normalized_status.as_str(),
                "succeeded" | "succeed" | "completed" | "done"
            ) {
                let completed = if a.provider == "fal" {
                    #[cfg(test)]
                    let result_url = self
                        .test_origin
                        .as_ref()
                        .map(|origin| {
                            origin
                                .join(&format!("upstream/fal/result/{}", submission.reference))
                                .unwrap()
                        })
                        .unwrap();
                    #[cfg(not(test))]
                    let result_url = Self::fal_result_url(&a.model, &submission.reference)?;
                    let result = self
                        .client
                        .get(result_url)
                        .header("authorization", self.authorization("fal")?)
                        .send()?;
                    if !result.status().is_success() {
                        // fal can mark a queue request COMPLETED yet reject the
                        // result payload with a documented 422 policy/model
                        // validation error. At this point the provider ref is
                        // authoritative, so close it non-rewardably rather
                        // than endlessly resuming the same result request.
                        if result.status().as_u16() == 422 {
                            return Err(Error::ProviderTerminal {
                                reason: "provider_failed",
                                http_status: None,
                                provider_code: None,
                            });
                        }
                        return Err(Error::Provider("fal_result".into()));
                    };
                    serde_json::from_slice::<Value>(&result.bytes()?)?
                } else {
                    value
                };
                if a.provider == "google-cloud" {
                    if let Some(bytes) = completed
                        .pointer("/response/videos/0/bytesBase64Encoded")
                        .and_then(Value::as_str)
                    {
                        let media = B64
                            .decode(bytes)
                            .map_err(|_| Error::Protocol("google_output_base64"))?;
                        if media.is_empty() || media.len() > MAX_MEDIA_BYTES {
                            return Err(Error::Protocol("invalid_output_bytes"));
                        };
                        return Ok(media);
                    }
                    let gcs = completed
                        .pointer("/response/videos/0/gcsUri")
                        .and_then(Value::as_str)
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    let path = gcs
                        .strip_prefix("gs://")
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    let (bucket, object) = path
                        .split_once('/')
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    if bucket
                        != self
                            .google_output_bucket
                            .as_deref()
                            .ok_or(Error::Protocol("google_cloud_project_location_or_bucket"))?
                        || object.is_empty()
                    {
                        return Err(Error::Protocol("google_output_bucket"));
                    }
                    #[cfg(not(test))]
                    let escaped: String = object
                        .bytes()
                        .map(|b| {
                            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
                                (b as char).to_string()
                            } else {
                                format!("%{b:02X}")
                            }
                        })
                        .collect();
                    #[cfg(test)]
                    let url = self
                        .test_origin
                        .as_ref()
                        .unwrap()
                        .join("upstream/google-cloud/output")
                        .unwrap();
                    #[cfg(not(test))] let url=reqwest::Url::parse(&format!("https://storage.googleapis.com/storage/v1/b/{bucket}/o/{escaped}?alt=media")).map_err(|_|Error::Protocol("google_output_uri"))?;
                    let output = self
                        .client
                        .get(url)
                        .header("authorization", self.authorization("google-cloud")?)
                        .send()?;
                    if !output.status().is_success() {
                        return Err(Error::Provider("google_output_download".into()));
                    };
                    return Self::media(output);
                }
                let url = self
                    .output_url(&a.provider, &completed)
                    .ok_or(Error::Protocol("provider_output_url"))?;
                let output = self.client.get(url).send()?;
                if !output.status().is_success() {
                    return Err(Error::Provider("output_download".into()));
                };
                return Self::media(output);
            }
            // Do not turn a near-expiry assignment into an unbounded polling
            // loop merely because the normal provider cadence is three seconds.
            std::thread::sleep(std::cmp::min(
                std::time::Duration::from_secs(3),
                deadline.saturating_duration_since(std::time::Instant::now()),
            ));
        }
    }
}
