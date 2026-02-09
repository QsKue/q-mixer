use std::io::{Cursor, Read, Seek, SeekFrom};

pub struct BytesSource {
    cursor: Cursor<Vec<u8>>,
    size: u64,
}

impl BytesSource {
    pub fn new(bytes: Vec<u8>) -> Self {
        let size = bytes.len() as u64;
        Self {
            cursor: Cursor::new(bytes),
            size,
        }
    }
}

impl super::AudioSource for BytesSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.cursor.read(buf)
    }
    
    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        self.cursor.seek(SeekFrom::Start(pos))
    }
    
    fn size(&self) -> Option<u64> {
        Some(self.size)
    }
    
    fn is_seekable(&self) -> bool {
        true
    }
}