use crate::context::document::Document;
use crate::context::document::IntoDocument;
use lopdf::{Document as PdfDoc, Object};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io::Error;
use std::path::{Path, PathBuf};

use super::FsDocumentError;

/// A pdf document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct PdfDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for PdfDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "pdf" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

impl IntoDocument for PdfDocument {
    type Error = FsDocumentError<lopdf::Error>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let path = &self.path;

        let doc = load_pdf(&self.path).await?;
        let title = doc
            .get_toc()
            .ok()
            .and_then(|toc| {
                toc.toc
                    .into_iter()
                    .min_by_key(|toc| toc.level)
                    .map(|toc| toc.title.to_string())
            })
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|os_str| os_str.to_str())
                    .unwrap_or("Untitled Document")
                    .to_string()
            });
        let text = get_pdf_text(&doc)?;
        if !text.errors.is_empty() {
            tracing::error!(
                "Encountered errors while extracting text from PDF at {path:?}: {:?}",
                text.errors
            );
        }

        let mut all_text = String::new();
        for paragraph in text.text.values().flatten() {
            all_text.push_str(paragraph);
            all_text.push('\n');
        }

        Ok(Document::from_parts(title, all_text))
    }
}

static IGNORE: &[&[u8]] = &[
    b"Length",
    b"BBox",
    b"FormType",
    b"Matrix",
    b"Type",
    b"XObject",
    b"Subtype",
    b"Filter",
    b"ColorSpace",
    b"Width",
    b"Height",
    b"BitsPerComponent",
    b"Length1",
    b"Length2",
    b"Length3",
    b"PTEX.FileName",
    b"PTEX.PageNumber",
    b"PTEX.InfoDict",
    b"FontDescriptor",
    b"ExtGState",
    b"MediaBox",
    b"Annot",
];

#[derive(Debug, Deserialize, Serialize)]
struct PdfText {
    text: BTreeMap<u32, Vec<String>>, // Key is page number
    errors: Vec<String>,
}

fn filter_func(object_id: (u32, u16), object: &mut Object) -> Option<((u32, u16), Object)> {
    if IGNORE.contains(&object.type_name().unwrap_or_default()) {
        return None;
    }
    if let Ok(d) = object.as_dict_mut() {
        d.remove(b"Producer");
        d.remove(b"ModDate");
        d.remove(b"Creator");
        d.remove(b"ProcSet");
        d.remove(b"Procset");
        d.remove(b"XObject");
        d.remove(b"MediaBox");
        d.remove(b"Annots");
        if d.is_empty() {
            return None;
        }
    }
    Some((object_id, object.to_owned()))
}

async fn load_pdf<P: AsRef<Path>>(path: P) -> Result<PdfDoc, Error> {
    PdfDoc::load_filtered(path, filter_func)
        .await
        .map_err(|e| Error::other(e.to_string()))
}

fn get_pdf_text(doc: &PdfDoc) -> Result<PdfText, Error> {
    let mut pdf_text: PdfText = PdfText {
        text: BTreeMap::new(),
        errors: Vec::new(),
    };
    let pages: Vec<Result<(u32, Vec<String>), Error>> = doc
        .get_pages()
        .into_iter()
        .map(
            |(page_num, page_id): (u32, (u32, u16))| -> Result<(u32, Vec<String>), Error> {
                let text = doc.extract_text(&[page_num]).map_err(|e| {
                    Error::other(format!(
                        "Failed to extract text from page {page_num} id={page_id:?}: {e:}"
                    ))
                })?;
                Ok((
                    page_num,
                    text.split('\n')
                        .map(|s| s.trim_end().to_string())
                        .collect::<Vec<String>>(),
                ))
            },
        )
        .collect();
    for page in pages {
        match page {
            Ok((page_num, lines)) => {
                pdf_text.text.insert(page_num, lines);
            }
            Err(e) => {
                pdf_text.errors.push(e.to_string());
            }
        }
    }
    Ok(pdf_text)
}
