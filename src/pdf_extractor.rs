use anyhow::{Result, Context};
use pdf_extract::extract_text;
use std::path::Path;
use tracing::{info, warn, error};

/// Simple PDF text extractor using pdf-extract crate
pub struct PdfExtractor;

impl PdfExtractor {
    /// Extract text from a PDF file
    pub fn extract_text_from_file<P: AsRef<Path>>(pdf_path: P) -> Result<String> {
        let path = pdf_path.as_ref();
        
        info!("Extracting text from PDF: {}", path.display());
        
        // Check if file exists
        if !path.exists() {
            return Err(anyhow::anyhow!("PDF file not found: {}", path.display()));
        }
        
        // Check file extension
        if let Some(ext) = path.extension() {
            if ext.to_string_lossy().to_lowercase() != "pdf" {
                warn!("File extension is not .pdf: {}", path.display());
            }
        }
        
        // Extract text using pdf-extract
        match extract_text(&path) {
            Ok(text) => {
                let char_count = text.len();
                info!("Successfully extracted {} characters from PDF", char_count);
                
                if text.trim().is_empty() {
                    warn!("Extracted text is empty - PDF might be scanned or contain only images");
                }
                
                Ok(text)
            }
            Err(e) => {
                error!("Failed to extract text from PDF: {}", e);
                Err(anyhow::anyhow!("PDF extraction failed: {}", e))
            }
        }
    }
    
    /// Extract text from PDF bytes
    pub fn extract_text_from_bytes(pdf_bytes: &[u8]) -> Result<String> {
        info!("Extracting text from PDF bytes ({} bytes)", pdf_bytes.len());
        
        // Write bytes to temporary file for pdf-extract
        use std::io::Write;
        let temp_path = std::env::temp_dir().join(format!("temp_pdf_{}.pdf", 
            std::process::id()));
        
        {
            let mut temp_file = std::fs::File::create(&temp_path)
                .context("Failed to create temporary PDF file")?;
            temp_file.write_all(pdf_bytes)
                .context("Failed to write PDF bytes to temporary file")?;
        }
        
        // Extract text from temporary file
        let result = Self::extract_text_from_file(&temp_path);
        
        // Clean up temporary file
        if let Err(e) = std::fs::remove_file(&temp_path) {
            warn!("Failed to remove temporary PDF file: {}", e);
        }
        
        result
    }
    
    /// Check if a file is likely a PDF based on its content
    pub fn is_pdf_content(content: &[u8]) -> bool {
        // Check for PDF magic bytes
        content.starts_with(b"%PDF")
    }
    
    /// Get basic metadata about the extraction
    pub fn get_extraction_metadata<P: AsRef<Path>>(pdf_path: P) -> Result<PdfMetadata> {
        let path = pdf_path.as_ref();
        let file_size = std::fs::metadata(path)?.len();
        
        let text = Self::extract_text_from_file(path)?;
        let char_count = text.len();
        let word_count = text.split_whitespace().count();
        let line_count = text.lines().count();
        
        Ok(PdfMetadata {
            file_size_bytes: file_size,
            character_count: char_count,
            word_count,
            line_count,
            has_content: char_count > 0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PdfMetadata {
    pub file_size_bytes: u64,
    pub character_count: usize,
    pub word_count: usize,
    pub line_count: usize,
    pub has_content: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_pdf_magic_bytes() {
        let pdf_content = b"%PDF-1.4\nsome content";
        assert!(PdfExtractor::is_pdf_content(pdf_content));
        
        let not_pdf = b"This is not a PDF";
        assert!(!PdfExtractor::is_pdf_content(not_pdf));
    }

    #[test]
    fn test_nonexistent_file() {
        let result = PdfExtractor::extract_text_from_file("/nonexistent/file.pdf");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_bytes() {
        let result = PdfExtractor::extract_text_from_bytes(&[]);
        assert!(result.is_err());
    }

    // Note: Testing actual PDF extraction would require a real PDF file
    // For integration tests, we would use the test PDF in uploads/
}