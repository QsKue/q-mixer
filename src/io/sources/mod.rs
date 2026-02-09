pub mod bytes;
pub mod file;

pub use bytes::BytesSource;
pub use file::FileSource;

pub trait AudioSource: Send + Sync {
    
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    
    fn seek(&mut self, pos: u64) -> std::io::Result<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "seek not supported"
        ))
    }
    
    fn size(&self) -> Option<u64> { None }
    
    fn is_seekable(&self) -> bool { false }
}