//! GGUF file parsing: header, metadata key-value table, tensor directory.
//!
//! The writer is kept for round-trip testing.

use fusor_ir::Result;
use fusor_ir::dtype::{Dtype, QFmt};
use fusor_ir::error::Error;
use smallvec::SmallVec;
use std::io::{Read, Seek, SeekFrom, Write};

/// GGUF wire format tags. Wider than [`QFmt`] because a file may name a
/// format fusor does not ingest; [`Self::to_qfmt`] is the total gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl GgmlType {
    pub const fn to_dtype(self) -> Option<Dtype> {
        match self {
            Self::F32 => Some(Dtype::F32),
            Self::F16 => Some(Dtype::F16),
            Self::Q4_0 => Some(Dtype::Q(QFmt::Q4_0)),
            Self::Q5_0 => Some(Dtype::Q(QFmt::Q5_0)),
            Self::Q8_0 => Some(Dtype::Q(QFmt::Q8_0)),
            Self::Q4K => Some(Dtype::Q(QFmt::Q4K)),
            Self::Q5K => Some(Dtype::Q(QFmt::Q5K)),
            Self::Q6K => Some(Dtype::Q(QFmt::Q6K)),
            _ => None,
        }
    }

    pub const fn to_qfmt(self) -> Option<QFmt> {
        match self.to_dtype() {
            Some(Dtype::Q(q)) => Some(q),
            _ => None,
        }
    }

    pub const fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            _ => return None,
        })
    }
}

/// `GGUF`. A byte-reversed spelling is also accepted.
pub const GGUF_MAGIC_BYTES: [u8; 4] = *b"GGUF";

/// Tensor data starts at the next multiple of this unless the file overrides
/// it with `general.alignment`.
pub const DEFAULT_ALIGNMENT: u64 = 32;

fn io<E: std::fmt::Display>(e: E) -> Error {
    Error::Io(e.to_string())
}

/// Container version. V1 length-prefixes with u32; V2 and V3 with u64.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufVersion {
    V1 = 1,
    V2 = 2,
    #[default]
    V3 = 3,
}

impl GgufVersion {
    pub const fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::V1,
            2 => Self::V2,
            3 => Self::V3,
            _ => return None,
        })
    }

    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(io)?;
        let forward = magic == GGUF_MAGIC_BYTES;
        let reversed = magic
            .iter()
            .rev()
            .zip(GGUF_MAGIC_BYTES.iter())
            .all(|(a, b)| a == b);
        if !forward && !reversed {
            return Err(Error::Io(format!("bad GGUF magic {magic:?}")));
        }
        let version = read_le_u32(reader)?;
        Self::from_u32(version)
            .ok_or_else(|| Error::Io(format!("unsupported GGUF version {version}")))
    }

    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_all(&GGUF_MAGIC_BYTES).map_err(io)?;
        write_le_u32(writer, self as u32)
    }
}

macro_rules! le_pair {
    ($read:ident, $write:ident, $ty:ty) => {
        #[doc = concat!("Read a little-endian `", stringify!($ty), "`.")]
        pub fn $read<R: Read>(reader: &mut R) -> Result<$ty> {
            let mut bytes = [0u8; std::mem::size_of::<$ty>()];
            reader.read_exact(&mut bytes).map_err(io)?;
            Ok(<$ty>::from_le_bytes(bytes))
        }
        #[doc = concat!("Write a little-endian `", stringify!($ty), "`.")]
        pub fn $write<W: Write>(writer: &mut W, value: $ty) -> Result<()> {
            writer.write_all(&value.to_le_bytes()).map_err(io)
        }
    };
}

le_pair!(read_le_u8, write_le_u8, u8);
le_pair!(read_le_i8, write_le_i8, i8);
le_pair!(read_le_u16, write_le_u16, u16);
le_pair!(read_le_i16, write_le_i16, i16);
le_pair!(read_le_u32, write_le_u32, u32);
le_pair!(read_le_i32, write_le_i32, i32);
le_pair!(read_le_u64, write_le_u64, u64);
le_pair!(read_le_i64, write_le_i64, i64);
le_pair!(read_le_f32, write_le_f32, f32);
le_pair!(read_le_f64, write_le_f64, f64);

/// Array and string lengths: u32 on V1, u64 on V2/V3.
pub fn read_array_length<R: Read>(reader: &mut R, version: GgufVersion) -> Result<usize> {
    Ok(match version {
        GgufVersion::V1 => read_le_u32(reader)? as usize,
        GgufVersion::V2 | GgufVersion::V3 => read_le_u64(reader)? as usize,
    })
}

fn write_array_length<W: Write>(writer: &mut W, version: GgufVersion, value: usize) -> Result<()> {
    match version {
        GgufVersion::V1 => write_le_u32(writer, value as u32),
        GgufVersion::V2 | GgufVersion::V3 => write_le_u64(writer, value as u64),
    }
}

/// A length-prefixed UTF-8 string. GGUF says non-null-terminated; in practice
/// producers emit trailing NULs and invalid UTF-8, so both are tolerated.
pub fn read_string<R: Read>(reader: &mut R, version: GgufVersion) -> Result<String> {
    let len = read_array_length(reader, version)?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes).map_err(io)?;
    while let Some(0) = bytes.last() {
        bytes.pop();
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn write_string<W: Write>(writer: &mut W, value: &str, version: GgufVersion) -> Result<()> {
    write_array_length(writer, version, value.len())?;
    writer.write_all(value.as_bytes()).map_err(io)
}

/// Wire tag of a metadata value. The numbering is not contiguous — U64/I64/F64
/// were appended after Bool/String/Array.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufMetadataValueType {
    pub const fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }
}

/// One metadata value, as the wire format types them.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Generate the widening accessors: every integer variant coerces to every
/// other integer type, floats to floats, anything else is a `Dtype` error.
macro_rules! widening {
    ($name:ident, $ty:ty, $variant:ident $(, $from:ident)* $(,)?) => {
        #[doc = concat!("Coerce to `", stringify!($ty), "`, widening across the numeric variants.")]
        pub fn $name(&self) -> Result<$ty> {
            match self {
                Self::$variant(v) => Ok(*v as $ty),
                $( Self::$from(v) => Ok(*v as $ty), )*
                other => Err(Error::Dtype(format!(
                    "metadata value {other:?} is not a {}", stringify!($ty)
                ))),
            }
        }
    };
}

impl GgufValue {
    pub const fn value_type(&self) -> GgufMetadataValueType {
        match self {
            Self::U8(_) => GgufMetadataValueType::U8,
            Self::I8(_) => GgufMetadataValueType::I8,
            Self::U16(_) => GgufMetadataValueType::U16,
            Self::I16(_) => GgufMetadataValueType::I16,
            Self::U32(_) => GgufMetadataValueType::U32,
            Self::I32(_) => GgufMetadataValueType::I32,
            Self::U64(_) => GgufMetadataValueType::U64,
            Self::I64(_) => GgufMetadataValueType::I64,
            Self::F32(_) => GgufMetadataValueType::F32,
            Self::F64(_) => GgufMetadataValueType::F64,
            Self::Bool(_) => GgufMetadataValueType::Bool,
            Self::String(_) => GgufMetadataValueType::String,
            Self::Array(_) => GgufMetadataValueType::Array,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    widening!(to_u8, u8, U8, U16, U32, U64, I8, I16, I32, I64);
    widening!(to_i8, i8, I8, I16, I32, I64, U8, U16, U32, U64);
    widening!(to_u16, u16, U16, U8, U32, U64, I8, I16, I32, I64);
    widening!(to_i16, i16, I16, I8, I32, I64, U8, U16, U32, U64);
    widening!(to_u32, u32, U32, U8, U16, U64, I8, I16, I32, I64);
    widening!(to_i32, i32, I32, I8, I16, I64, U8, U16, U32, U64);
    widening!(to_u64, u64, U64, U8, U16, U32, I8, I16, I32, I64);
    widening!(to_i64, i64, I64, I8, I16, I32, U8, U16, U32, U64);
    widening!(to_f32, f32, F32, F64);
    widening!(to_f64, f64, F64, F32);

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            other => Err(Error::Dtype(format!(
                "metadata value {other:?} is not a bool"
            ))),
        }
    }

    /// The string body, or a `Dtype` error. Named `to_string_value` because
    /// `to_string` belongs to `Display`.
    pub fn to_string_value(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(Error::Dtype(format!(
                "metadata value {other:?} is not a string"
            ))),
        }
    }

    pub fn to_array(&self) -> Result<&[GgufValue]> {
        match self {
            Self::Array(a) => Ok(a),
            other => Err(Error::Dtype(format!(
                "metadata value {other:?} is not an array"
            ))),
        }
    }

    fn read<R: Read>(
        reader: &mut R,
        ty: GgufMetadataValueType,
        version: GgufVersion,
    ) -> Result<Self> {
        Ok(match ty {
            GgufMetadataValueType::U8 => Self::U8(read_le_u8(reader)?),
            GgufMetadataValueType::I8 => Self::I8(read_le_i8(reader)?),
            GgufMetadataValueType::U16 => Self::U16(read_le_u16(reader)?),
            GgufMetadataValueType::I16 => Self::I16(read_le_i16(reader)?),
            GgufMetadataValueType::U32 => Self::U32(read_le_u32(reader)?),
            GgufMetadataValueType::I32 => Self::I32(read_le_i32(reader)?),
            GgufMetadataValueType::U64 => Self::U64(read_le_u64(reader)?),
            GgufMetadataValueType::I64 => Self::I64(read_le_i64(reader)?),
            GgufMetadataValueType::F32 => Self::F32(read_le_f32(reader)?),
            GgufMetadataValueType::F64 => Self::F64(read_le_f64(reader)?),
            GgufMetadataValueType::Bool => match read_le_u8(reader)? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                other => return Err(Error::Io(format!("invalid bool byte {other}"))),
            },
            GgufMetadataValueType::String => Self::String(read_string(reader, version)?),
            GgufMetadataValueType::Array => {
                let tag = read_le_u32(reader)?;
                let element = GgufMetadataValueType::from_u32(tag)
                    .ok_or_else(|| Error::Io(format!("invalid array value type {tag}")))?;
                let len = read_array_length(reader, version)?;
                let mut values = Vec::with_capacity(len.min(1 << 16));
                for _ in 0..len {
                    values.push(GgufValue::read(reader, element, version)?);
                }
                Self::Array(values)
            }
        })
    }

    fn write<W: Write>(&self, writer: &mut W, version: GgufVersion) -> Result<()> {
        match self {
            Self::U8(v) => write_le_u8(writer, *v),
            Self::I8(v) => write_le_i8(writer, *v),
            Self::U16(v) => write_le_u16(writer, *v),
            Self::I16(v) => write_le_i16(writer, *v),
            Self::U32(v) => write_le_u32(writer, *v),
            Self::I32(v) => write_le_i32(writer, *v),
            Self::U64(v) => write_le_u64(writer, *v),
            Self::I64(v) => write_le_i64(writer, *v),
            Self::F32(v) => write_le_f32(writer, *v),
            Self::F64(v) => write_le_f64(writer, *v),
            Self::Bool(v) => write_le_u8(writer, u8::from(*v)),
            Self::String(s) => write_string(writer, s, version),
            Self::Array(values) => {
                let first = values
                    .first()
                    .ok_or_else(|| Error::Io("cannot write an empty GGUF array".into()))?;
                write_le_u32(writer, first.value_type() as u32)?;
                write_array_length(writer, version, values.len())?;
                for v in values {
                    v.write(writer, version)?;
                }
                Ok(())
            }
        }
    }
}

/// Elements one block of this wire type holds.
pub const fn ggml_block_elements(ty: GgmlType) -> u64 {
    match ty {
        GgmlType::F32 | GgmlType::F16 => 1,
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q8_1 => 32,
        GgmlType::Q2K
        | GgmlType::Q3K
        | GgmlType::Q4K
        | GgmlType::Q5K
        | GgmlType::Q6K
        | GgmlType::Q8K => 256,
    }
}

/// On-disk bytes one block of this wire type occupies. Covers the formats
/// fusor cannot ingest too, because the tensor directory must stay walkable
/// past a tensor it will later refuse.
pub const fn ggml_block_bytes(ty: GgmlType) -> u64 {
    match ty {
        GgmlType::F32 => 4,
        GgmlType::F16 => 2,
        GgmlType::Q4_0 => 18,
        GgmlType::Q4_1 => 20,
        GgmlType::Q5_0 => 22,
        GgmlType::Q5_1 => 24,
        GgmlType::Q8_0 => 34,
        GgmlType::Q8_1 => 36,
        GgmlType::Q2K => 84,
        GgmlType::Q3K => 110,
        GgmlType::Q4K => 144,
        GgmlType::Q5K => 176,
        GgmlType::Q6K => 210,
        GgmlType::Q8K => 292,
    }
}

/// The total ingest gate: F32, F16 and the six block formats fusor decodes
/// end to end. Everything else is an error naming the tag.
pub fn ingest_qfmt(ty: GgmlType) -> Result<Dtype> {
    ty.to_dtype()
        .ok_or_else(|| Error::Dtype(format!("gguf type {ty:?} has no ingest path")))
}

/// One tensor's directory entry. `shape` is **reversed at read**: GGUF stores
/// dimensions fastest-varying first, fusor uses row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub ty: GgmlType,
    pub shape: SmallVec<[u64; 4]>,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
    pub bytes: u64,
}

impl GgufTensor {
    /// Storage bytes for a shape in this wire type, rejecting a shape that is
    /// not a whole number of blocks.
    pub fn byte_size(ty: GgmlType, shape: &[u64]) -> Result<u64> {
        let elements: u64 = shape.iter().product();
        let per_block = ggml_block_elements(ty);
        if !elements.is_multiple_of(per_block) {
            return Err(Error::Shape(format!(
                "{elements} elements is not a multiple of {ty:?}'s block size {per_block}"
            )));
        }
        Ok(elements / per_block * ggml_block_bytes(ty))
    }
}

/// Header, key-value table and tensor directory of one GGUF file.
///
/// `entries` and `tensors` are ordered: the write path must reproduce the
/// read order for a byte-stable round trip.
#[derive(Clone, Debug, Default)]
pub struct GgufMetadata {
    pub version: GgufVersion,
    pub entries: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensor>,
    pub tensor_data_offset: u64,
}

impl GgufMetadata {
    /// Look up a metadata value.
    ///
    /// A leading `.` makes the lookup a **suffix match** and the shortest
    /// matching key wins, so `.attention.head_count` resolves the
    /// architecture-prefixed `qwen3.attention.head_count` without the caller
    /// knowing the architecture.
    pub fn get_value(&self, key: &str) -> Option<&GgufValue> {
        if key.starts_with('.') {
            self.entries
                .iter()
                .filter(|(k, _)| k.ends_with(key))
                .min_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
        } else {
            self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
    }

    /// `general.architecture`.
    pub fn architecture(&self) -> Option<&str> {
        self.get_value("general.architecture")?.as_str()
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Alignment the file declares, defaulting to [`DEFAULT_ALIGNMENT`].
    pub fn alignment(&self) -> u64 {
        match self
            .get_value("general.alignment")
            .and_then(|v| v.to_u64().ok())
        {
            Some(a) if a > 0 => a,
            _ => DEFAULT_ALIGNMENT,
        }
    }

    /// Read the header, the key-value table and the tensor directory.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let version = GgufVersion::read(reader)?;
        let tensor_count = read_array_length(reader, version)?;
        let metadata_count = read_array_length(reader, version)?;

        let mut entries = Vec::with_capacity(metadata_count.min(1 << 16));
        for _ in 0..metadata_count {
            let key = read_string(reader, version)?;
            let tag = read_le_u32(reader)?;
            let ty = GgufMetadataValueType::from_u32(tag)
                .ok_or_else(|| Error::Io(format!("invalid metadata value type {tag}")))?;
            entries.push((key, GgufValue::read(reader, ty, version)?));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 16));
        for _ in 0..tensor_count {
            let name = read_string(reader, version)?;
            let rank = read_le_u32(reader)? as usize;
            let mut shape: SmallVec<[u64; 4]> = SmallVec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_array_length(reader, version)? as u64);
            }
            // GGUF is fastest-varying-first; fusor is row-major.
            shape.reverse();
            let tag = read_le_u32(reader)?;
            let ty = GgmlType::from_u32(tag)
                .ok_or_else(|| Error::Dtype(format!("unknown ggml type {tag}")))?;
            let offset = read_le_u64(reader)?;
            let bytes = GgufTensor::byte_size(ty, &shape)?;
            tensors.push(GgufTensor {
                name,
                ty,
                shape,
                offset,
                bytes,
            });
        }

        let position = reader.stream_position().map_err(io)?;
        let mut this = Self {
            version,
            entries,
            tensors,
            tensor_data_offset: 0,
        };
        let alignment = this.alignment();
        this.tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(this)
    }

    /// Write the header, table, directory and tensor payloads. Kept only so
    /// [`Self::read`] can be round-trip tested.
    pub fn write<'a, W: Write + Seek>(
        &self,
        writer: &mut W,
        tensors: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> Result<()> {
        self.version.write(writer)?;
        write_array_length(writer, self.version, self.tensors.len())?;
        write_array_length(writer, self.version, self.entries.len())?;

        for (key, value) in &self.entries {
            write_string(writer, key, self.version)?;
            write_le_u32(writer, value.value_type() as u32)?;
            value.write(writer, self.version)?;
        }

        for tensor in &self.tensors {
            write_string(writer, &tensor.name, self.version)?;
            write_le_u32(writer, tensor.shape.len() as u32)?;
            for dim in tensor.shape.iter().rev() {
                write_array_length(writer, self.version, *dim as usize)?;
            }
            write_le_u32(writer, tensor.ty as u32)?;
            write_le_u64(writer, tensor.offset)?;
        }

        let position = writer.stream_position().map_err(io)?;
        let alignment = self.alignment();
        let data_offset = position.div_ceil(alignment) * alignment;
        // Materialize the alignment padding: a seek past the end does not
        // extend an in-memory cursor.
        writer
            .write_all(&vec![0u8; (data_offset - position) as usize])
            .map_err(io)?;

        for (name, bytes) in tensors {
            let tensor = self
                .tensor(name)
                .ok_or_else(|| Error::Io(format!("unknown tensor {name}")))?;
            writer
                .seek(SeekFrom::Start(data_offset + tensor.offset))
                .map_err(io)?;
            writer.write_all(bytes).map_err(io)?;
        }
        Ok(())
    }
}

enum Backing {
    Mapped(memmap2::Mmap),
    Owned(Box<[u8]>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m,
            Self::Owned(b) => b,
        }
    }
}

/// A GGUF file, memory-mapped or in memory. Tensor bytes are borrowed straight
/// out of the backing, so loading a weight is a slice, not a copy.
pub struct Gguf {
    backing: Backing,
    metadata: GgufMetadata,
}

impl Gguf {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).map_err(io)?;
        // SAFETY: mapping a file the caller named. The usual mmap caveat
        // applies — concurrent truncation by another process would fault.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(io)?;
        Self::with_backing(Backing::Mapped(map))
    }

    /// In-memory variant, for tests and for callers that already hold the file
    /// (a browser fetch, an embedded asset).
    pub fn from_bytes(bytes: impl Into<Box<[u8]>>) -> Result<Self> {
        Self::with_backing(Backing::Owned(bytes.into()))
    }

    fn with_backing(backing: Backing) -> Result<Self> {
        let metadata = {
            let mut cursor = std::io::Cursor::new(backing.bytes());
            GgufMetadata::read(&mut cursor)?
        };
        Ok(Self { backing, metadata })
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn bytes(&self) -> &[u8] {
        self.backing.bytes()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.metadata.tensors.iter().map(|t| t.name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.metadata.tensor(name)
    }

    /// Raw bytes of one tensor, borrowed from the backing.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Error::Io(format!("unknown tensor {name}")))?;
        let start = (self.metadata.tensor_data_offset + tensor.offset) as usize;
        let end = start + tensor.bytes as usize;
        let all = self.backing.bytes();
        if end > all.len() {
            return Err(Error::Io(format!(
                "tensor {name} runs to {end} but the file is {} bytes",
                all.len()
            )));
        }
        Ok(&all[start..end])
    }
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    /// Build a synthetic GGUF file in memory. Tensors are laid out back to
    /// back from offset 0 in the order given.
    pub(crate) fn build(
        version: GgufVersion,
        entries: &[(&str, GgufValue)],
        tensors: &[(&str, GgmlType, &[u64], Vec<u8>)],
    ) -> Vec<u8> {
        let mut infos = Vec::new();
        let mut offset = 0u64;
        for (name, ty, shape, _) in tensors {
            let shape: SmallVec<[u64; 4]> = shape.iter().copied().collect();
            let bytes = GgufTensor::byte_size(*ty, &shape).unwrap();
            infos.push(GgufTensor {
                name: (*name).to_string(),
                ty: *ty,
                shape,
                offset,
                bytes,
            });
            offset += bytes;
        }
        let meta = GgufMetadata {
            version,
            entries: entries
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
            tensors: infos,
            tensor_data_offset: 0,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        meta.write(
            &mut buf,
            tensors.iter().map(|(n, _, _, d)| (*n, d.as_slice())),
        )
        .unwrap();
        buf.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_header_round_trip() {
        let file = fixture::build(
            GgufVersion::V3,
            &[
                ("general.architecture", GgufValue::String("qwen3".into())),
                ("qwen3.block_count", GgufValue::U32(28)),
            ],
            &[
                ("token_embd.weight", GgmlType::F32, &[2, 3], vec![0u8; 24]),
                (
                    "blk.0.attn_q.weight",
                    GgmlType::Q4_0,
                    &[1, 32],
                    (0..18u8).collect(),
                ),
                ("output_norm.weight", GgmlType::F16, &[4], vec![0u8; 8]),
            ],
        );
        let gguf = Gguf::from_bytes(file).unwrap();
        let meta = gguf.metadata();
        assert_eq!(meta.version, GgufVersion::V3);
        assert_eq!(meta.architecture(), Some("qwen3"));
        assert_eq!(
            meta.get_value("qwen3.block_count")
                .unwrap()
                .to_u64()
                .unwrap(),
            28
        );
        assert_eq!(meta.tensor_data_offset % 32, 0);
        assert_eq!(meta.alignment(), 32);

        let t = meta.tensor("token_embd.weight").unwrap();
        assert_eq!(t.shape.as_slice(), &[2, 3]);
        assert_eq!(t.offset, 0);
        assert_eq!(t.bytes, 24);
        let q = meta.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.ty, GgmlType::Q4_0);
        assert_eq!(q.shape.as_slice(), &[1, 32]);
        assert_eq!(q.bytes, 18);
        assert_eq!(q.offset, 24);
        assert_eq!(
            gguf.tensor_bytes("blk.0.attn_q.weight").unwrap(),
            (0..18u8).collect::<Vec<_>>().as_slice()
        );
        assert_eq!(gguf.tensor_names().count(), 3);

        // A declared alignment is honoured.
        let file = fixture::build(
            GgufVersion::V3,
            &[
                ("general.alignment", GgufValue::U32(64)),
                ("general.architecture", GgufValue::String("qwen3".into())),
            ],
            &[("a", GgmlType::F32, &[1], vec![1, 2, 3, 4])],
        );
        let gguf = Gguf::from_bytes(file).unwrap();
        assert_eq!(gguf.metadata().alignment(), 64);
        assert_eq!(gguf.metadata().tensor_data_offset % 64, 0);
        assert_eq!(gguf.tensor_bytes("a").unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn v1_and_v2_length_prefixes_round_trip() {
        for version in [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3] {
            let file = fixture::build(
                version,
                &[("general.architecture", GgufValue::String("llama".into()))],
                &[("a", GgmlType::F16, &[8], vec![7u8; 16])],
            );
            let gguf = Gguf::from_bytes(file).unwrap();
            assert_eq!(gguf.metadata().version, version);
            assert_eq!(gguf.metadata().architecture(), Some("llama"));
            assert_eq!(gguf.tensor_bytes("a").unwrap(), [7u8; 16]);
        }
    }
}
