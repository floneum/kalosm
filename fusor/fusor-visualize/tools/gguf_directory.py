"""Read GGUF v2/v3 tensor descriptors without loading tensor payloads."""
import struct


def directory(data):
    pos = 0

    def read(fmt):
        nonlocal pos
        value = struct.unpack_from('<' + fmt, data, pos)
        pos += struct.calcsize('<' + fmt)
        return value[0] if len(value) == 1 else value

    def string():
        nonlocal pos
        length = read('Q')
        assert length <= len(data) - pos
        value = data[pos:pos + length].decode('utf-8')
        pos += length
        return value

    def metadata(tag):
        nonlocal pos
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                   6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if tag in formats:
            return read(formats[tag])
        if tag == 8:
            return string()
        assert tag == 9, f'unsupported GGUF metadata tag {tag}'
        element, count = read('I'), read('Q')
        if element in formats:
            pos += struct.calcsize(formats[element]) * count
            assert pos <= len(data)
        else:
            for _ in range(count):
                metadata(element)

    assert data[:4] == b'GGUF'
    pos = 4
    assert read('I') in (2, 3), 'only GGUF v2/v3 supported'
    tensors, entries = read('Q'), read('Q')
    alignment = 32
    for _ in range(entries):
        key = string()
        value = metadata(read('I'))
        if key == 'general.alignment':
            alignment = value
    records = []
    for _ in range(tensors):
        name = string()
        rank = read('I')
        assert 0 < rank <= 8
        shape = [read('Q') for _ in range(rank)][::-1]
        records.append(dict(name=name, shape=shape, tag=read('I'), offset=read('Q')))
    assert alignment > 0
    return (pos + alignment - 1) // alignment * alignment, records
