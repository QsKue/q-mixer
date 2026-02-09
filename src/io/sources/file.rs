use std::{io::{Read, Seek, SeekFrom}, path::Path};

pub struct FileSource {
    file: std::fs::File,
    size: u64,
}

impl FileSource {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(&path)?;
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }
}

impl super::AudioSource for FileSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
    
    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        self.file.seek(SeekFrom::Start(pos))
    }
    
    fn size(&self) -> Option<u64> {
        Some(self.size)
    }
    
    fn is_seekable(&self) -> bool {
        true
    }
}