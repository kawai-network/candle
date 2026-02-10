package candle

import (
	"fmt"
	"os"
	"path/filepath"

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
