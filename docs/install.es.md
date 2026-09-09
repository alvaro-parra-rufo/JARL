# Instalación

**Español** · [English](install.md)

JARL se ejecuta en Linux. En Windows debe utilizarse mediante WSL2; la ejecución nativa desde PowerShell o `cmd` no está soportada.

## 📋 Requisitos

- Linux x86-64 o Windows con WSL2 y una distribución Linux, como Ubuntu.
- Git y GNU Make.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) para instalar Python y las dependencias.
- Python 3.12 o posterior. `uv` puede instalar una versión compatible durante la sincronización.
- Una GPU NVIDIA y un controlador compatible con CUDA 13 para los entrenamientos acelerados.

El uso de las funciones agénticas requiere además un modelo local servido por Ollama o acceso a una API compatible con OpenAI. El servidor MCP es opcional y necesita un cliente compatible.

## 🐧 Preparar WSL2

Si trabajas desde Windows, instala WSL2 y ejecuta el resto de comandos dentro de la terminal Linux. JARL no debe instalarse desde PowerShell.

Comprueba primero que WSL puede acceder a la GPU:

```bash
nvidia-smi
```

Este comando debe mostrar la GPU y el controlador de NVIDIA. El proyecto instala `jax[cuda13]` y las dependencias CUDA correspondientes mediante `uv`; el controlador del sistema debe ser compatible con CUDA 13. En WSL, la GPU se expone mediante el controlador instalado en Windows.

## 📦 Instalar JARL

Instala `uv` dentro de Linux o WSL si todavía no está disponible:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Abre una terminal nueva y comprueba la instalación:

```bash
uv --version
```

Clona el repositorio e instala el proyecto:

```bash
git clone https://github.com/alvaro-parra-rufo/JARL.git
cd JARL
make install
```

`make install` crea el entorno con `uv`, sincroniza las dependencias e instala los hooks de pre-commit. Por este motivo, debe ejecutarse dentro de un repositorio clonado con Git y no desde un archivo ZIP.

## ✅ Verificar la instalación

Comprueba el backend y los dispositivos detectados por JAX:

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Para entrenar con la GPU, la primera línea debe indicar `gpu` y la lista debe contener al menos un dispositivo CUDA. Las comprobaciones generales del proyecto se ejecutan con:

```bash
make check
make test
```

Runner Lab se inicia desde la raíz del repositorio:

```bash
make app
```

## 🤖 Configurar un modelo de lenguaje

La configuración del modelo solo es necesaria para utilizar las funciones agénticas. Copia primero la plantilla de variables locales:

```bash
cp .env.example .env
```

El archivo `.env` está ignorado por Git.

### Ollama local

Ollama no necesita una API key. Debe estar en ejecución y tener descargado el modelo configurado:

```bash
ollama pull qwen2.5:7b
```

```dotenv
JARL_LLM_PROVIDER=ollama
JARL_LLM_MODEL=qwen2.5:7b
JARL_LLM_BASE_URL=http://127.0.0.1:11434
```

Si Ollama se ejecuta en Windows y JARL dentro de WSL, la dirección de loopback puede no alcanzar el servicio. En ese caso, configura `JARL_LLM_BASE_URL` con una dirección del host accesible desde WSL.

### API compatible con OpenAI

Para utilizar un proveedor remoto hacen falta la URL base, el nombre del modelo y una API key:

```dotenv
JARL_LLM_PROVIDER=openai_compatible
JARL_LLM_BASE_URL=https://api.example.com/v1
JARL_LLM_MODEL=provider-model-name
JARL_LLM_API_KEY=replace-with-your-api-key
```

JARL utiliza el identificador `openai_compatible` para cualquier proveedor que implemente este tipo de API. No utilices el nombre comercial del proveedor como valor de `JARL_LLM_PROVIDER`.

## 🔌 Utilizar MCP

El servidor MCP permite operar los experimentos desde un cliente compatible, como Cursor, Codex o Claude Code. El cliente inicia JARL mediante `stdio` y debe indicar en `JARL_MCP_EXPERIMENT_DIR` el directorio del experimento activo.

La configuración disponible para Cursor se encuentra en `.cursor/mcp.example.json`. Consulta la [guía del servidor MCP](development/mcp.md) para preparar el cliente y arrancar el servidor.

## 🛠️ Problemas frecuentes

| Problema | Comprobación |
|---|---|
| JAX utiliza `cpu` | Ejecuta `nvidia-smi` dentro de WSL y revisa el controlador de NVIDIA. |
| JAX no puede cargar CUDA | Comprueba que el controlador sea compatible con CUDA 13 y evita mezclar instalaciones CUDA distintas. |
| Ollama no responde desde WSL | Utiliza una dirección del host accesible desde WSL en `JARL_LLM_BASE_URL`. |
| Falta la API key | Define `JARL_LLM_API_KEY` junto con la URL y el modelo del proveedor. |
