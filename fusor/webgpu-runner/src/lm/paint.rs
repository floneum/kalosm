//! Encode attention and embedding grids as BMP data URIs.
//! Rows are bottom-up BGR pixels padded to four bytes.

/// A `width x height` grid of `(r, g, b)` as a `data:image/bmp;base64` URI.
pub fn data_uri(pixels: &[[u8; 3]], width: usize, height: usize) -> String {
    let mut uri = String::from("data:image/bmp;base64,");
    base64_into(&encode_bmp(pixels, width, height), &mut uri);
    uri
}

fn encode_bmp(pixels: &[[u8; 3]], width: usize, height: usize) -> Vec<u8> {
    debug_assert_eq!(pixels.len(), width * height);
    let stride = (width * 3).next_multiple_of(4);
    let pixel_bytes = stride * height;
    let mut bmp = Vec::with_capacity(54 + pixel_bytes);

    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&((54 + pixel_bytes) as u32).to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
    bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
    bmp.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(height as i32).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
    bmp.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    bmp.extend_from_slice(&(pixel_bytes as u32).to_le_bytes());
    bmp.extend_from_slice(&2835i32.to_le_bytes()); // 72 dpi, x
    bmp.extend_from_slice(&2835i32.to_le_bytes()); // 72 dpi, y
    bmp.extend_from_slice(&0u32.to_le_bytes()); // palette entries
    bmp.extend_from_slice(&0u32.to_le_bytes()); // important colors

    // Bottom-up: the last row of the grid is the first row of the file.
    for y in (0..height).rev() {
        let row = &pixels[y * width..(y + 1) * width];
        for [r, g, b] in row {
            bmp.extend_from_slice(&[*b, *g, *r]);
        }
        bmp.resize(54 + (height - y) * stride, 0);
    }

    bmp
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_into(bytes: &[u8], out: &mut String) {
    out.reserve(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let [a, b, c] = [
            group[0],
            group.get(1).copied().unwrap_or(0),
            group.get(2).copied().unwrap_or(0),
        ];
        let word = (u32::from(a) << 16) | (u32::from(b) << 8) | u32::from(c);
        for shift in [18, 12, 6, 0] {
            out.push(char::from(ALPHABET[((word >> shift) & 0x3f) as usize]));
        }
        // The tail is padded, and each absent input byte costs one output
        // character: three bytes are four characters, two are three plus `=`.
        let pad = 3 - group.len();
        out.truncate(out.len() - pad);
        for _ in 0..pad {
            out.push('=');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_rows_are_bottom_up_bgr_with_row_padding() {
        let pixels = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ];
        let bmp = encode_bmp(&pixels, 3, 2);
        assert_eq!(
            &bmp[54..],
            &[
                3, 2, 1, 6, 5, 4, 9, 8, 7, 0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(
            u32::from_le_bytes(bmp[2..6].try_into().unwrap()) as usize,
            bmp.len()
        );
    }

    #[test]
    fn base64_matches_the_rfc_test_vectors() {
        let mut out = String::new();
        base64_into(b"", &mut out);
        assert_eq!(out, "");
        for (input, expected) in [
            (&b"f"[..], "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ] {
            let mut out = String::new();
            base64_into(input, &mut out);
            assert_eq!(out, expected, "base64 of {input:?}");
        }
    }
}
