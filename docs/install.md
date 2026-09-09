# Installation

[Español](install.es.md) · **English**

JARL runs on Linux. Windows users need WSL2 because native execution from PowerShell or `cmd` is not supported.

## 📋 Requirements

- Linux x86-64, or Windows with WSL2 and a Linux distribution such as Ubuntu.
- Git and GNU Make.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) to install Python and project dependencies.
- Python 3.12 or newer. `uv` can install a compatible version during synchronization.
- An NVIDIA GPU and a driver compatible with CUDA 13 for accelerated training.

Agentic features also need either a local model served by Ollama or access to an OpenAI-compatible API. The MCP server is optional and requires a compatible client.

## 🐧 Set up WSL2

On Windows, install WSL2 and run the remaining commands from the Linux terminal. Do not install JARL from PowerShell.

Check that WSL can access the GPU:

```bash
nvidia-smi
```

This command should display the NVIDIA GPU and its driver. The project installs `jax[cuda13]` and the corresponding CUDA dependencies through `uv`, so the system driver must support CUDA 13. Under WSL, the Windows driver exposes the GPU to Linux.

## 📦 Install JARL

Install `uv` inside Linux or WSL if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal and check the installation:

```bash
uv --version
```

Clone the repository and install the project:

```bash
git clone https://github.com/alvaro-parra-rufo/JARL.git
cd JARL
make install
```

`make install` creates the environment with `uv`, synchronizes the dependencies, and installs the pre-commit hooks. The pre-commit step needs Git metadata and will fail inside an extracted ZIP archive.

## ✅ Verify the installation

Check the backend and devices detected by JAX:

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

For GPU training, the first line should report `gpu`, and the list should contain at least one CUDA device. Run the project checks with:

```bash
make check
make test
```

Start Runner Lab from the repository root:

```bash
make app
```

## 🤖 Configure a language model

You only need a language model for the agentic features. Copy the local environment template:

```bash
cp .env.example .env
```

Git ignores `.env`. Keep credentials out of files that will be shared or committed.

### Local Ollama model

Ollama does not require an API key. Its service must be running, and the configured model must be available locally:

```bash
ollama pull qwen2.5:7b
```

```dotenv
JARL_LLM_PROVIDER=ollama
JARL_LLM_MODEL=qwen2.5:7b
JARL_LLM_BASE_URL=http://127.0.0.1:11434
```

When Ollama runs on Windows and JARL runs inside WSL, the loopback address may not reach the service. Set `JARL_LLM_BASE_URL` to a host address that WSL can access.

### OpenAI-compatible API

A remote provider requires a base URL, model name, and API key:

```dotenv
JARL_LLM_PROVIDER=openai_compatible
JARL_LLM_BASE_URL=https://api.example.com/v1
JARL_LLM_MODEL=provider-model-name
JARL_LLM_API_KEY=replace-with-your-api-key
```

Use `openai_compatible` as `JARL_LLM_PROVIDER` for any provider that implements this API format. Provider brand names are not valid values for this setting.

## 🔌 Use MCP

The MCP server exposes JARL experiments to compatible clients such as Cursor, Codex, or Claude Code. The client starts JARL over `stdio` and sets `JARL_MCP_EXPERIMENT_DIR` to the active experiment directory.

The repository includes a Cursor template at `.cursor/mcp.example.json`. See the [MCP server guide](development/mcp.md) for the client configuration and startup command.

## 🛠️ Troubleshooting

| Problem | Check |
|---|---|
| JAX uses `cpu` | Run `nvidia-smi` inside WSL and check the NVIDIA driver. |
| JAX cannot load CUDA | Check that the driver supports CUDA 13 and remove conflicting CUDA installations. |
| `make install` cannot install pre-commit | Make sure you cloned the repository with Git. |
| Ollama does not respond from WSL | Set `JARL_LLM_BASE_URL` to a host address accessible from WSL. |
| The API key is missing | Set `JARL_LLM_API_KEY` along with the provider URL and model. |
