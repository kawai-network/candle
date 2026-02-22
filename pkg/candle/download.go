package candle

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
)

// ModelFiles holds paths to downloaded model artifacts.
type ModelFiles struct {
	// Model weights (safetensors format)
	ModelPath string
	// Model configuration
	ConfigPath string
	// Tokenizer (tokenizer.json, tokenizer.model, or vocab.txt)
	TokenizerPath string
	// Optional: merges.txt for BPE tokenizers
	MergesPath string
	// Optional: generation_config.json
	GenerationConfigPath string
	// Optional: preprocessor_config.json (for vision models)
	PreprocessorConfigPath string
}

// DownloadModel downloads model artifacts from Hugging Face Hub.
// repoID: e.g. "sentence-transformers/all-MiniLM-L6-v2"
// cacheDir: directory to store the model. If empty, uses ~/.cache/huggingface/hub
//
// Returns a ModelFiles struct with paths to each downloaded file.
// Files that don't exist for a given model will have empty paths.
func DownloadModel(repoID, cacheDir string) (*ModelFiles, error) {
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get user home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".cache", "huggingface", "hub")
	}

	repo := hub.New(repoID).WithCacheDir(cacheDir)

	files := &ModelFiles{}

	// Download config.json (always required)
	configPath, err := repo.DownloadFile("config.json")
	if err != nil {
		return nil, fmt.Errorf("failed to download config.json: %w", err)
	}
	files.ConfigPath = configPath

	// Download model weights (safetensors format)
	// Try single file first, then sharded
	if modelPath, err := repo.DownloadFile("model.safetensors"); err == nil {
		files.ModelPath = modelPath
	} else {
		// Try sharded format - download the index first
		if indexPath, err2 := repo.DownloadFile("model.safetensors.index.json"); err2 == nil {
			files.ModelPath = indexPath
		}
		// Some models use different naming: pytorch_model.bin, etc.
		// We prefer safetensors but fall back
	}

	// Download tokenizer - try multiple formats
	if tokPath, err := repo.DownloadFile("tokenizer.json"); err == nil {
		files.TokenizerPath = tokPath
	} else if tokPath, err := repo.DownloadFile("tokenizer.model"); err == nil {
		files.TokenizerPath = tokPath
	} else if tokPath, err := repo.DownloadFile("vocab.txt"); err == nil {
		files.TokenizerPath = tokPath
	}

	// Optional: merges.txt for BPE tokenizers
	if mergesPath, err := repo.DownloadFile("merges.txt"); err == nil {
		files.MergesPath = mergesPath
	}

	// Optional: generation_config.json
	if genPath, err := repo.DownloadFile("generation_config.json"); err == nil {
		files.GenerationConfigPath = genPath
	}

	// Optional: preprocessor_config.json
	if prepPath, err := repo.DownloadFile("preprocessor_config.json"); err == nil {
		files.PreprocessorConfigPath = prepPath
	}

	return files, nil
}

const (
	DefaultLibraryVersion = "v0.1.0"
	DefaultLibraryRepo    = "kawai-network/candle"
)

func getLibraryName() (string, error) {
	goOS := runtime.GOOS
	goArch := runtime.GOARCH

	switch goOS {
	case "darwin":
		switch goArch {
		case "arm64":
			return "libcandle_binding-darwin-arm64", nil
		case "amd64":
			return "libcandle_binding-darwin-amd64", nil
		default:
			return "", fmt.Errorf("unsupported darwin architecture: %s", goArch)
		}
	case "linux":
		switch goArch {
		case "amd64":
			return "libcandle_binding-linux-amd64", nil
		case "arm64":
			return "libcandle_binding-linux-arm64", nil
		default:
			return "", fmt.Errorf("unsupported linux architecture: %s", goArch)
		}
	case "windows":
		switch goArch {
		case "amd64":
			return "libcandle_binding-windows-amd64", nil
		case "arm64":
			return "libcandle_binding-windows-arm64", nil
		default:
			return "", fmt.Errorf("unsupported windows architecture: %s", goArch)
		}
	default:
		return "", fmt.Errorf("unsupported platform: %s/%s", goOS, goArch)
	}
}

func getLibraryExtension() string {
	switch runtime.GOOS {
	case "darwin":
		return "dylib"
	case "windows":
		return "dll"
	default:
		return "so"
	}
}

type LibraryVariant string

const (
	LibraryBasic LibraryVariant = "basic"
	LibraryVideo LibraryVariant = "video"
)

func DownloadLibrary(variant LibraryVariant, version, cacheDir string) (string, error) {
	if version == "" {
		version = DefaultLibraryVersion
	}

	libName, err := getLibraryName()
	if err != nil {
		return "", err
	}
	ext := getLibraryExtension()

	filename := fmt.Sprintf("%s-%s.%s.gz", libName, variant, ext)

	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get user home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".cache", "go-candle", "libs", version)
	}

	destPath := filepath.Join(cacheDir, strings.TrimSuffix(filename, ".gz"))

	if _, err := os.Stat(destPath); err == nil {
		return destPath, nil
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache dir: %w", err)
	}

	downloadURL := fmt.Sprintf(
		"https://github.com/%s/releases/download/%s/%s",
		DefaultLibraryRepo, version, filename,
	)

	resp, err := http.Get(downloadURL)
	if err != nil {
		return "", fmt.Errorf("failed to download library: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download library: HTTP %d", resp.StatusCode)
	}

	gzPath := destPath + ".gz"
	out, err := os.Create(gzPath)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		return "", fmt.Errorf("failed to write library: %w", err)
	}
	out.Close()

	gzFile, err := os.Open(gzPath)
	if err != nil {
		return "", fmt.Errorf("failed to open compressed file: %w", err)
	}
	defer gzFile.Close()

	gzReader, err := gzip.NewReader(gzFile)
	if err != nil {
		return "", fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	finalOut, err := os.Create(destPath)
	if err != nil {
		return "", fmt.Errorf("failed to create library file: %w", err)
	}
	defer finalOut.Close()

	if _, err := io.Copy(finalOut, gzReader); err != nil {
		return "", fmt.Errorf("failed to decompress library: %w", err)
	}

	if err := os.Chmod(destPath, 0755); err != nil {
		return "", fmt.Errorf("failed to set permissions: %w", err)
	}

	os.Remove(gzPath)

	return destPath, nil
}

func EnsureVideoLibrary(version string) (string, error) {
	libPath := os.Getenv("CANDLE_LIB_PATH")
	if libPath != "" {
		if _, err := os.Stat(libPath); err == nil {
			return libPath, nil
		}
	}

	cacheDir := ""
	if version == "" {
		version = DefaultLibraryVersion
	}

	libName, err := getLibraryName()
	if err != nil {
		return "", err
	}
	ext := getLibraryExtension()

	home, _ := os.UserHomeDir()
	if home != "" {
		cacheDir = filepath.Join(home, ".cache", "go-candle", "libs", version)
		cachedPath := filepath.Join(cacheDir, libName+"-video."+ext)
		if _, err := os.Stat(cachedPath); err == nil {
			return cachedPath, nil
		}
	}

	return DownloadLibrary(LibraryVideo, version, "")
}
