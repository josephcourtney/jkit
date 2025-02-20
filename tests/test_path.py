import os
import time
from pathlib import Path

import pytest

from jkit.path import normalize_path_input, unpropagated_changes


def test_directory_expansion(tmp_path):
    """Test directory inputs are expanded into contained files."""
    # Target directory with old file
    target_dir = tmp_path / "target_dir"
    target_dir.mkdir()
    target_file = target_dir / "file.txt"
    target_file.touch()
    old_time = time.time() - 100
    os.utime(target_file, (old_time, old_time))

    # Source directory with new file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "file.txt"
    source_file.touch()
    new_time = time.time()
    os.utime(source_file, (new_time, new_time))

    assert unpropagated_changes(target_dir, source_dir) is True


def test_all_sources_older(tmp_path):
    """Test when all sources are older than the oldest target."""
    # Older sources
    source = tmp_path / "source.txt"
    source.touch()
    old_time = time.time() - 100
    os.utime(source, (old_time, old_time))

    # Newer targets
    target = tmp_path / "target.txt"
    target.touch()
    new_time = time.time()
    os.utime(target, (new_time, new_time))

    assert unpropagated_changes(target, source) is False


def test_one_source_newer(tmp_path):
    """Test when at least one source is newer than the oldest target."""
    # Targets with old mtime
    target1 = tmp_path / "target1.txt"
    target1.touch()
    old_time = time.time() - 100
    os.utime(target1, (old_time, old_time))

    target2 = tmp_path / "target2.txt"
    target2.touch()
    os.utime(target2, (old_time + 50, old_time + 50))

    # Mix of old and new sources
    source_old = tmp_path / "source_old.txt"
    source_old.touch()
    os.utime(source_old, (old_time - 50, old_time - 50))

    source_new = tmp_path / "source_new.txt"
    source_new.touch()
    new_time = time.time()
    os.utime(source_new, (new_time, new_time))

    assert unpropagated_changes([target1, target2], [source_old, source_new]) is True


def test_glob_patterns(tmp_path, monkeypatch):
    """Test glob patterns in source and target inputs."""
    monkeypatch.chdir(tmp_path)  # Ensure glob resolves correctly

    # Target files matched by glob
    (tmp_path / "target_a.txt").touch()
    (tmp_path / "target_b.txt").touch()
    old_time = time.time() - 100
    for f in tmp_path.glob("target_*.txt"):
        os.utime(f, (old_time, old_time))

    # New source files matched by glob
    (tmp_path / "source_1.txt").touch()
    (tmp_path / "source_2.txt").touch()
    new_time = time.time()
    for f in tmp_path.glob("source_*.txt"):
        os.utime(f, (new_time, new_time))

    assert unpropagated_changes("target_*.txt", "source_*.txt") is True


def test_same_mtime(tmp_path):
    """Test when newest source and oldest target have same mtime."""
    target = tmp_path / "target.txt"
    target.touch()
    source = tmp_path / "source.txt"
    source.touch()
    same_time = time.time()
    os.utime(target, (same_time, same_time))
    os.utime(source, (same_time, same_time))

    assert unpropagated_changes(target, source) is False


def test_mixed_path_types(tmp_path):
    """Test mixed Path/str/iterable inputs."""
    # Target as Path object
    target = tmp_path / "target.txt"
    target.touch()
    os.utime(target, (time.time() - 100, time.time() - 100))

    # Sources as mixed types
    source1 = tmp_path / "source1.txt"
    source1.touch()
    source2 = tmp_path / "source2.txt"
    source2.touch()
    new_time = time.time()
    os.utime(source1, (new_time, new_time))
    os.utime(source2, (new_time, new_time))

    input_sources = [source1, str(source2), ["nested", ["*.txt"]]]
    assert unpropagated_changes(target, input_sources) is True


def test_empty_inputs():
    """Test empty iterables for sources or targets."""
    assert unpropagated_changes([], []) is False
    assert unpropagated_changes("non_existent.txt", []) is False
    assert unpropagated_changes([], "non_existent.txt") is False


def test_single_existing_path(tmp_path, monkeypatch):
    """Test single Path object pointing to existing file."""
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "test.txt"
    test_file.touch()

    result = list(normalize_path_input(test_file))
    assert len(result) == 1
    assert result[0] == test_file.resolve()


def test_recursive_glob(tmp_path, monkeypatch):
    """Test recursive glob pattern."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "root.txt").touch()
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "nested.txt").touch()

    result = list(normalize_path_input("**/*.txt"))
    resolved_tmp = tmp_path.resolve()
    expected = {resolved_tmp / "root.txt", resolved_tmp / "sub/nested.txt"}
    assert set(result) == expected


def test_nested_iterables(tmp_path, monkeypatch):
    """Test nested iterables with mixed types (no duplicates)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "iter.txt").touch()
    (tmp_path / "data.csv").touch()
    subdir = tmp_path / "nested"
    subdir.mkdir()
    (subdir / "deep.txt").touch()

    input_arg = [Path("iter.txt"), "data.csv", ["nested/*.txt", "nonexistent.file"], [["*.csv"]]]

    result = set(normalize_path_input(input_arg))
    resolved_tmp = tmp_path.resolve()
    expected = {
        resolved_tmp / "iter.txt",
        resolved_tmp / "data.csv",
        resolved_tmp / "nested/deep.txt",
    }
    assert result == expected


def test_non_existing_paths(tmp_path):
    """Test paths that don't exist are skipped."""
    result = list(normalize_path_input(tmp_path / "ghost.txt"))
    assert not result
    result = list(normalize_path_input("phantom.csv"))
    assert not result


def test_absolute_paths(tmp_path, monkeypatch):
    """Test absolute path handling."""
    monkeypatch.chdir(tmp_path)
    abs_file = tmp_path / "absolute_test.txt"
    abs_file.touch()

    # Absolute Path object
    result = list(normalize_path_input(abs_file))
    assert result[0] == abs_file.resolve()

    # Absolute string path
    result = list(normalize_path_input(str(abs_file)))
    assert result[0] == abs_file.resolve()


def test_edge_cases():
    """Test empty iterables and invalid types."""
    assert not list(normalize_path_input([]))
    with pytest.raises(TypeError):
        list(normalize_path_input(123))


def test_complex_mixed_case(tmp_path, monkeypatch):
    """Test combination of valid inputs without duplicates."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mix1.txt").touch()
    (tmp_path / "mix2.csv").touch()
    subdir = tmp_path / "mixed"
    subdir.mkdir()
    (subdir / "deep_mix.txt").touch()

    input_arg = [Path("mix1.txt"), "mix2.csv", ["mixed/*.txt", Path("non_existent.dir")]]

    result = set(normalize_path_input(input_arg))
    resolved_tmp = tmp_path.resolve()
    expected = {
        resolved_tmp / "mix1.txt",
        resolved_tmp / "mix2.csv",
        resolved_tmp / "mixed/deep_mix.txt",
    }
    assert result == expected


def test_no_source_files(tmp_path):
    """Test when there are no valid source files."""
    target = tmp_path / "target.txt"
    target.touch()
    # Source points to non-existent file
    assert unpropagated_changes(target, tmp_path / "ghost.txt") is False


def test_no_target_files(tmp_path):
    """Test when there are no valid target files."""
    source = tmp_path / "source.txt"
    source.touch()
    # Target points to non-existent file
    assert unpropagated_changes(tmp_path / "ghost.txt", source) is False


def test_hidden_files(tmp_path):
    hidden_file = tmp_path / ".hidden"
    hidden_file.touch()
    result = list(normalize_path_input(hidden_file))
    assert result == [hidden_file.resolve()]


def test_symlinks(tmp_path):
    target = tmp_path / "target.txt"
    target.touch()
    symlink = tmp_path / "symlink.txt"
    symlink.symlink_to(target)
    result = list(normalize_path_input(symlink))
    assert result == [target.resolve()]


def test_unicode_paths(tmp_path):
    unicode_file = tmp_path / " Café ☕.txt"
    unicode_file.touch()
    result = list(normalize_path_input(unicode_file))
    assert result == [unicode_file.resolve()]


def test_permission_denied(tmp_path):
    restricted_file = tmp_path / "restricted.txt"
    restricted_file.touch()
    restricted_file.chmod(0o000)  # No permissions
    try:
        result = list(normalize_path_input(restricted_file))
        assert not result  # Should skip inaccessible files
    finally:
        # Reset permissions to allow cleanup
        restricted_file.chmod(0o644)


def test_invalid_glob():
    result = list(normalize_path_input("invalid[glob"))
    assert not result  # Should handle invalid globs gracefully


def test_concurrent_modifications(tmp_path):
    target = tmp_path / "target.txt"
    target.touch()
    source = tmp_path / "source.txt"
    source.touch()
    # Simulate concurrent modification
    source.write_text("updated")
    assert unpropagated_changes(target, source) is True


def test_mixed_file_types(tmp_path):
    target_file = tmp_path / "target.txt"
    target_file.touch()
    target_dir = tmp_path / "target_dir"
    target_dir.mkdir()
    source_file = tmp_path / "source.txt"
    source_file.touch()
    assert unpropagated_changes([target_file, target_dir], source_file) is True


def test_cross_platform_paths(tmp_path):
    target = tmp_path / "target.txt"
    target.touch()
    source = tmp_path / "source.txt"
    source.touch()
    # Simulate cross-platform path handling
    assert unpropagated_changes(str(target), str(source)) is True


def test_file_system_errors(tmp_path):
    inaccessible_dir = tmp_path / "inaccessible"
    inaccessible_dir.mkdir()
    inaccessible_dir.chmod(0o000)  # No permissions
    try:
        source = tmp_path / "source.txt"
        source.touch()
        assert unpropagated_changes(inaccessible_dir, source) is False
    finally:
        # Reset permissions to allow cleanup
        inaccessible_dir.chmod(0o700)


def test_complex_setup(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("base")
    restricted_file = base_dir / "restricted.txt"
    restricted_file.touch()
    restricted_file.chmod(0o000)  # No permissions
    try:
        result = list(normalize_path_input(restricted_file))
        assert not result  # Should skip inaccessible files
    finally:
        # Reset permissions to allow cleanup
        restricted_file.chmod(0o644)
