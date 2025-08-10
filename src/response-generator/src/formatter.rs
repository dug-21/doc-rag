//! Response formatting system supporting multiple output formats

use crate::error::{Result, ResponseError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

/// Response formatter for multiple output formats
#[derive(Debug, Clone)]
pub struct ResponseFormatter {
    /// Configuration for formatting
    config: FormatterConfig,
    
    /// Custom format templates
    templates: HashMap<String, String>,
}

/// Configuration for response formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatterConfig {
    /// Default output format
    pub default_format: OutputFormat,
    
    /// Enable syntax highlighting for code blocks
    pub enable_syntax_highlighting: bool,
    
    /// Maximum line length for text wrapping
    pub max_line_length: usize,
    
    /// Indent size for structured formats
    pub indent_size: usize,
    
    /// Include metadata in output
    pub include_metadata: bool,
    
    /// Custom CSS styles for HTML output
    pub custom_css: Option<String>,
}

/// Supported output formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Plain text format
    Text,
    
    /// JSON format with structured data
    Json,
    
    /// Markdown format with rich text
    Markdown,
    
    /// HTML format with styling
    Html,
    
    /// XML format for structured data
    Xml,
    
    /// YAML format for configuration-like output
    Yaml,
    
    /// CSV format for tabular data
    Csv,
    
    /// Custom format using templates
    Custom(String),
}

/// Formatted response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedResponse {
    /// The formatted content
    pub content: String,
    
    /// Output format used
    pub format: OutputFormat,
    
    /// Content type/MIME type
    pub content_type: String,
    
    /// Character encoding
    pub encoding: String,
    
    /// Content length in bytes
    pub content_length: usize,
    
    /// Additional formatting metadata
    pub metadata: HashMap<String, String>,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            default_format: OutputFormat::Json,
            enable_syntax_highlighting: true,
            max_line_length: 80,
            indent_size: 2,
            include_metadata: true,
            custom_css: None,
        }
    }
}

impl ResponseFormatter {
    /// Create a new response formatter
    pub fn new(config: FormatterConfig) -> Self {
        let mut formatter = Self {
            config,
            templates: HashMap::new(),
        };
        
        formatter.load_default_templates();
        formatter
    }

    /// Format response content
    #[instrument(skip(self, content), fields(format = ?format))]
    pub async fn format(&self, content: &str, format: OutputFormat) -> Result<String> {
        debug!("Formatting content ({} chars) to {:?}", content.len(), format);

        let formatted = match format {
            OutputFormat::Text => self.format_text(content).await?,
            OutputFormat::Json => self.format_json(content).await?,
            OutputFormat::Markdown => self.format_markdown(content).await?,
            OutputFormat::Html => self.format_html(content).await?,
            OutputFormat::Xml => self.format_xml(content).await?,
            OutputFormat::Yaml => self.format_yaml(content).await?,
            OutputFormat::Csv => self.format_csv(content).await?,
            OutputFormat::Custom(template_name) => self.format_custom(content, &template_name).await?,
        };

        Ok(formatted)
    }

    /// Format with full response metadata
    #[instrument(skip(self, content))]
    pub async fn format_with_metadata(
        &self,
        content: &str,
        format: OutputFormat,
        additional_metadata: Option<HashMap<String, String>>,
    ) -> Result<FormattedResponse> {
        let formatted_content = self.format(content, format.clone()).await?;
        let content_type = self.get_content_type(&format);
        let content_length = formatted_content.as_bytes().len();
        
        let mut metadata = HashMap::new();
        if let Some(meta) = additional_metadata {
            metadata.extend(meta);
        }
        
        if self.config.include_metadata {
            metadata.insert("formatter_version".to_string(), "1.0.0".to_string());
            metadata.insert("formatted_at".to_string(), chrono::Utc::now().to_rfc3339());
            metadata.insert("original_length".to_string(), content.len().to_string());
        }

        Ok(FormattedResponse {
            content: formatted_content,
            format,
            content_type,
            encoding: "UTF-8".to_string(),
            content_length,
            metadata,
        })
    }

    /// Add custom format template
    pub fn add_template(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
    }

    /// Get available format templates
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    /// Format as plain text
    async fn format_text(&self, content: &str) -> Result<String> {
        let mut formatted = content.to_string();
        
        // Apply text wrapping if configured
        if self.config.max_line_length > 0 {
            formatted = self.wrap_text(&formatted, self.config.max_line_length);
        }

        // Clean up extra whitespace
        formatted = self.normalize_whitespace(&formatted);

        Ok(formatted)
    }

    /// Format as JSON
    async fn format_json(&self, content: &str) -> Result<String> {
        let json_response = serde_json::json!({
            "response": content,
            "metadata": {
                "formatted_at": chrono::Utc::now().to_rfc3339(),
                "format": "json"
            }
        });

        if self.config.indent_size > 0 {
            serde_json::to_string_pretty(&json_response)
                .map_err(|e| ResponseError::formatting("json", e.to_string()))
        } else {
            serde_json::to_string(&json_response)
                .map_err(|e| ResponseError::formatting("json", e.to_string()))
        }
    }

    /// Format as Markdown
    async fn format_markdown(&self, content: &str) -> Result<String> {
        let mut markdown = String::new();

        // Split content into paragraphs and sentences for better formatting
        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        
        for (i, paragraph) in paragraphs.iter().enumerate() {
            if i > 0 {
                markdown.push_str("\n\n");
            }

            // Check if paragraph should be formatted as a list, code block, etc.
            let formatted_paragraph = self.format_markdown_paragraph(paragraph)?;
            markdown.push_str(&formatted_paragraph);
        }

        // Add metadata section if configured
        if self.config.include_metadata {
            markdown.push_str("\n\n---\n");
            markdown.push_str("*Generated by Response Generator*\n");
            markdown.push_str(&format!("*Formatted at: {}*", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        }

        Ok(markdown)
    }

    /// Format as HTML
    async fn format_html(&self, content: &str) -> Result<String> {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str("  <title>Response</title>\n");
        
        // Add custom CSS if provided
        if let Some(css) = &self.config.custom_css {
            html.push_str("  <style>\n");
            html.push_str(css);
            html.push_str("\n  </style>\n");
        } else {
            // Default CSS
            html.push_str("  <style>\n");
            html.push_str("    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 2rem; }\n");
            html.push_str("    .response-content { max-width: 800px; }\n");
            html.push_str("    .metadata { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ccc; font-size: 0.9em; color: #666; }\n");
            html.push_str("  </style>\n");
        }
        
        html.push_str("</head>\n<body>\n");
        html.push_str("  <div class=\"response-content\">\n");

        // Convert content to HTML paragraphs
        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        for paragraph in paragraphs {
            if !paragraph.trim().is_empty() {
                let escaped = html_escape::encode_text(paragraph);
                html.push_str(&format!("    <p>{}</p>\n", escaped));
            }
        }

        html.push_str("  </div>\n");

        // Add metadata section if configured
        if self.config.include_metadata {
            html.push_str("  <div class=\"metadata\">\n");
            html.push_str("    <p><em>Generated by Response Generator</em></p>\n");
            html.push_str(&format!("    <p><em>Formatted at: {}</em></p>\n", 
                          chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
            html.push_str("  </div>\n");
        }

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Format as XML
    async fn format_xml(&self, content: &str) -> Result<String> {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<response>\n");
        
        // Escape XML content
        let escaped_content = xml_escape::encode_text(content);
        
        // Split into paragraphs for better structure
        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        
        xml.push_str("  <content>\n");
        for (i, paragraph) in paragraphs.iter().enumerate() {
            if !paragraph.trim().is_empty() {
                let escaped = xml_escape::encode_text(paragraph.trim());
                xml.push_str(&format!("    <paragraph id=\"{}\">{}</paragraph>\n", i + 1, escaped));
            }
        }
        xml.push_str("  </content>\n");

        // Add metadata if configured
        if self.config.include_metadata {
            xml.push_str("  <metadata>\n");
            xml.push_str("    <generator>Response Generator</generator>\n");
            xml.push_str(&format!("    <formatted_at>{}</formatted_at>\n", 
                          chrono::Utc::now().to_rfc3339()));
            xml.push_str(&format!("    <content_length>{}</content_length>\n", content.len()));
            xml.push_str("  </metadata>\n");
        }

        xml.push_str("</response>");

        Ok(xml)
    }

    /// Format as YAML
    async fn format_yaml(&self, content: &str) -> Result<String> {
        let mut data = serde_yaml::Mapping::new();
        
        data.insert(
            serde_yaml::Value::String("response".to_string()),
            serde_yaml::Value::String(content.to_string())
        );

        if self.config.include_metadata {
            let mut metadata = serde_yaml::Mapping::new();
            metadata.insert(
                serde_yaml::Value::String("generator".to_string()),
                serde_yaml::Value::String("Response Generator".to_string())
            );
            metadata.insert(
                serde_yaml::Value::String("formatted_at".to_string()),
                serde_yaml::Value::String(chrono::Utc::now().to_rfc3339())
            );
            metadata.insert(
                serde_yaml::Value::String("content_length".to_string()),
                serde_yaml::Value::Number(serde_yaml::Number::from(content.len()))
            );
            
            data.insert(
                serde_yaml::Value::String("metadata".to_string()),
                serde_yaml::Value::Mapping(metadata)
            );
        }

        let yaml_value = serde_yaml::Value::Mapping(data);
        serde_yaml::to_string(&yaml_value)
            .map_err(|e| ResponseError::formatting("yaml", e.to_string()))
    }

    /// Format as CSV (for structured data)
    async fn format_csv(&self, content: &str) -> Result<String> {
        let mut csv = String::from("content,metadata\n");
        
        // Escape CSV content
        let escaped_content = self.escape_csv_field(content);
        let timestamp = chrono::Utc::now().to_rfc3339();
        let escaped_timestamp = self.escape_csv_field(&timestamp);
        
        csv.push_str(&format!("{},{}\n", escaped_content, escaped_timestamp));

        Ok(csv)
    }

    /// Format using custom template
    async fn format_custom(&self, content: &str, template_name: &str) -> Result<String> {
        let template = self.templates.get(template_name)
            .ok_or_else(|| ResponseError::formatting("custom", 
                format!("Template '{}' not found", template_name).as_str()))?;

        let mut formatted = template.clone();
        
        // Replace placeholders
        formatted = formatted.replace("{content}", content);
        formatted = formatted.replace("{timestamp}", &chrono::Utc::now().to_rfc3339());
        formatted = formatted.replace("{date}", &chrono::Utc::now().format("%Y-%m-%d").to_string());
        formatted = formatted.replace("{time}", &chrono::Utc::now().format("%H:%M:%S").to_string());
        formatted = formatted.replace("{length}", &content.len().to_string());

        Ok(formatted)
    }

    /// Get content type for format
    fn get_content_type(&self, format: &OutputFormat) -> String {
        match format {
            OutputFormat::Text => "text/plain".to_string(),
            OutputFormat::Json => "application/json".to_string(),
            OutputFormat::Markdown => "text/markdown".to_string(),
            OutputFormat::Html => "text/html".to_string(),
            OutputFormat::Xml => "application/xml".to_string(),
            OutputFormat::Yaml => "application/x-yaml".to_string(),
            OutputFormat::Csv => "text/csv".to_string(),
            OutputFormat::Custom(_) => "text/plain".to_string(),
        }
    }

    /// Load default format templates
    fn load_default_templates(&mut self) {
        // Simple template for basic formatting
        self.templates.insert(
            "simple".to_string(),
            "Response: {content}\nGenerated: {timestamp}".to_string()
        );

        // Email-like template
        self.templates.insert(
            "email".to_string(),
            "Subject: Response\n\n{content}\n\n---\nGenerated by Response Generator on {date} at {time}".to_string()
        );

        // Report template
        self.templates.insert(
            "report".to_string(),
            "# Response Report\n\nGenerated: {timestamp}\nContent Length: {length} characters\n\n## Response\n\n{content}".to_string()
        );
    }

    /// Wrap text to specified line length
    fn wrap_text(&self, text: &str, max_length: usize) -> String {
        let mut wrapped = String::new();
        
        for line in text.lines() {
            if line.len() <= max_length {
                wrapped.push_str(line);
                wrapped.push('\n');
            } else {
                let words: Vec<&str> = line.split_whitespace().collect();
                let mut current_line = String::new();
                
                for word in words {
                    if current_line.len() + word.len() + 1 > max_length {
                        if !current_line.is_empty() {
                            wrapped.push_str(&current_line);
                            wrapped.push('\n');
                            current_line.clear();
                        }
                    }
                    
                    if !current_line.is_empty() {
                        current_line.push(' ');
                    }
                    current_line.push_str(word);
                }
                
                if !current_line.is_empty() {
                    wrapped.push_str(&current_line);
                    wrapped.push('\n');
                }
            }
        }

        wrapped.trim_end().to_string()
    }

    /// Normalize whitespace in text
    fn normalize_whitespace(&self, text: &str) -> String {
        // Remove excessive whitespace while preserving paragraph breaks
        let lines: Vec<&str> = text.lines().collect();
        let mut normalized = String::new();
        let mut blank_line_count = 0;

        for line in lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                blank_line_count += 1;
                if blank_line_count <= 2 {
                    normalized.push('\n');
                }
            } else {
                blank_line_count = 0;
                normalized.push_str(trimmed);
                normalized.push('\n');
            }
        }

        normalized.trim_end().to_string()
    }

    /// Format a markdown paragraph
    fn format_markdown_paragraph(&self, paragraph: &str) -> Result<String> {
        let trimmed = paragraph.trim();
        
        // Check for list items
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") || 
           trimmed.chars().next().map_or(false, |c| c.is_numeric()) {
            return Ok(trimmed.to_string());
        }

        // Check for code blocks
        if trimmed.starts_with("```") || trimmed.starts_with("    ") {
            return Ok(trimmed.to_string());
        }

        // Check for headers
        if trimmed.starts_with('#') {
            return Ok(trimmed.to_string());
        }

        // Regular paragraph - apply emphasis if needed
        let formatted = self.apply_markdown_emphasis(trimmed);
        Ok(formatted)
    }

    /// Apply markdown emphasis to text
    fn apply_markdown_emphasis(&self, text: &str) -> String {
        let mut formatted = text.to_string();
        
        // Simple emphasis patterns (basic implementation)
        // In a real implementation, this would be more sophisticated
        
        // Bold for strong statements
        if text.contains("important") || text.contains("critical") || text.contains("essential") {
            formatted = formatted.replace("important", "**important**");
            formatted = formatted.replace("critical", "**critical**");
            formatted = formatted.replace("essential", "**essential**");
        }

        formatted
    }

    /// Escape CSV field
    fn escape_csv_field(&self, field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }
}

impl Default for ResponseFormatter {
    fn default() -> Self {
        Self::new(FormatterConfig::default())
    }
}

// Helper modules for XML and HTML escaping
mod html_escape {
    pub fn encode_text(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
}

mod xml_escape {
    pub fn encode_text(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_formatter_creation() {
        let formatter = ResponseFormatter::default();
        assert_eq!(formatter.config.default_format, OutputFormat::Json);
    }

    #[tokio::test]
    async fn test_text_formatting() {
        let formatter = ResponseFormatter::default();
        let content = "This is a test response.";
        
        let formatted = formatter.format(content, OutputFormat::Text).await.unwrap();
        assert_eq!(formatted, content);
    }

    #[tokio::test]
    async fn test_json_formatting() {
        let formatter = ResponseFormatter::default();
        let content = "Test response";
        
        let formatted = formatter.format(content, OutputFormat::Json).await.unwrap();
        assert!(formatted.contains("Test response"));
        assert!(formatted.contains("response"));
        
        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&formatted).unwrap();
        assert!(parsed["response"].is_string());
    }

    #[tokio::test]
    async fn test_markdown_formatting() {
        let formatter = ResponseFormatter::default();
        let content = "This is a test response.\n\nThis is a second paragraph.";
        
        let formatted = formatter.format(content, OutputFormat::Markdown).await.unwrap();
        assert!(formatted.contains("test response"));
        assert!(formatted.contains("second paragraph"));
    }

    #[tokio::test]
    async fn test_html_formatting() {
        let formatter = ResponseFormatter::default();
        let content = "Test & response with <tags>";
        
        let formatted = formatter.format(content, OutputFormat::Html).await.unwrap();
        assert!(formatted.contains("&amp;"));
        assert!(formatted.contains("&lt;tags&gt;"));
        assert!(formatted.contains("<!DOCTYPE html>"));
    }

    #[tokio::test]
    async fn test_custom_template() {
        let mut formatter = ResponseFormatter::default();
        formatter.add_template(
            "test_template".to_string(),
            "Content: {content}\nLength: {length}".to_string()
        );
        
        let content = "Test content";
        let formatted = formatter.format(content, OutputFormat::Custom("test_template".to_string())).await.unwrap();
        
        assert!(formatted.contains("Content: Test content"));
        assert!(formatted.contains("Length: 12"));
    }

    #[tokio::test]
    async fn test_format_with_metadata() {
        let formatter = ResponseFormatter::default();
        let content = "Test response";
        
        let formatted_response = formatter.format_with_metadata(
            content, 
            OutputFormat::Json, 
            None
        ).await.unwrap();
        
        assert_eq!(formatted_response.format, OutputFormat::Json);
        assert_eq!(formatted_response.content_type, "application/json");
        assert!(formatted_response.content_length > 0);
        assert!(!formatted_response.metadata.is_empty());
    }

    #[test]
    fn test_text_wrapping() {
        let formatter = ResponseFormatter::default();
        let long_text = "This is a very long line that should be wrapped at the specified length to improve readability.";
        
        let wrapped = formatter.wrap_text(long_text, 20);
        let lines: Vec<&str> = wrapped.lines().collect();
        
        assert!(lines.len() > 1);
        for line in lines {
            assert!(line.len() <= 25); // Some flexibility for word boundaries
        }
    }
}