use docx_rs::{Document as DocxFile, FromXML};
use std::path::PathBuf;

use std::fs::File;

use crate::context::document::{Document, IntoDocument};

use super::FsDocumentError;

/// A docx document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct DocxDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for DocxDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "docx" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

impl IntoDocument for DocxDocument {
    type Error = FsDocumentError<docx_rs::ReaderError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let file = File::open(self.path)?;
        let reader = std::io::BufReader::new(file);
        let docx = DocxFile::from_xml(reader).map_err(FsDocumentError::Decode)?;
        let mut text = String::new();
        for section in docx.children {
            match section {
                docx_rs::DocumentChild::Paragraph(paragraph) => {
                    for child in paragraph.children {
                        match child {
                            docx_rs::ParagraphChild::Run(run) => {
                                for child in run.children {
                                    match child {
                                        docx_rs::RunChild::Text(text_child) => {
                                            text += &text_child.text;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Document::from_parts("", text))
    }
}
