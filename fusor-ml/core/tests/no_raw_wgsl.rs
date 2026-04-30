use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

#[test]
fn workspace_sources_do_not_contain_legacy_raw_shader_text() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source_roots = [
        workspace_root.join("fusor-ml/core/src"),
        workspace_root.join("fusor-ml/fusor/src"),
        workspace_root.join("fusor-ml/gguf/src"),
        workspace_root.join("prototypes/phase-token-prototype/src"),
        workspace_root.join("prototypes/phase-token-prototype/examples"),
    ];
    let mut sources = Vec::new();
    for source_root in source_roots {
        rust_sources(&source_root, &mut sources);
    }
    sources.push(workspace_root.join("prototypes/phase-token-prototype/Cargo.toml"));
    let forbidden = [
        (
            ["ShaderSource::", "W", "g", "s", "l"].concat(),
            "submits raw shader text directly to wgpu",
        ),
        (
            ["naga::back::", "w", "g", "s", "l"].concat(),
            "still lowers through the raw shader-text backend",
        ),
        (
            ["FUSOR_DUMP_", "W", "G", "S", "L"].concat(),
            "still exposes the old raw shader dump hook",
        ),
        (
            ["W", "g", "s", "l"].concat(),
            "still uses the old shader-named storage or builder vocabulary",
        ),
        (
            ["w", "g", "s", "l"].concat(),
            "still mentions raw shader text generation",
        ),
        (
            ["W", "G", "S", "L"].concat(),
            "still mentions raw shader text generation",
        ),
        (
            [
                "P", "A", "S", "S", "T", "H", "R", "O", "U", "G", "H", "_", "S", "H", "A", "D",
                "E", "R", "S",
            ]
            .concat(),
            "still requests raw shader passthrough",
        ),
        (
            [
                "ShaderModuleDescriptor",
                "P",
                "a",
                "s",
                "s",
                "t",
                "h",
                "r",
                "o",
                "u",
                "g",
                "h",
            ]
            .concat(),
            "still submits raw shader text directly to wgpu",
        ),
        (
            [
                "create_shader_module_",
                "p",
                "a",
                "s",
                "s",
                "t",
                "h",
                "r",
                "o",
                "u",
                "g",
                "h",
            ]
            .concat(),
            "still submits raw shader text directly to wgpu",
        ),
        (
            ["naga::back::", "m", "s", "l"].concat(),
            "still lowers through the raw shader-text backend",
        ),
        (
            ["m", "s", "l", "-out"].concat(),
            "still enables raw shader text output",
        ),
        (
            ["M", "S", "L"].concat(),
            "still mentions raw shader text generation",
        ),
        (
            ["m", "s", "l"].concat(),
            "still mentions raw shader text generation",
        ),
    ];

    for path in sources {
        let source = fs::read_to_string(&path).unwrap();
        for (token, reason) in &forbidden {
            assert!(!source.contains(token), "{} {}", path.display(), reason);
        }
    }
}
