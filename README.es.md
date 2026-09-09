<div align="center">

![Logo de JARL](docs/assets/images/logo_JARL_light.png#gh-light-mode-only)
![Logo de JARL](docs/assets/images/logo_JARL_dark.png#gh-dark-mode-only)

</div>

---

<p align="center"><strong>Judgment-Augmented Reinforcement Learning</strong></p>

<p align="center"><strong>Español</strong> · <a href="README.md">English</a></p>

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat-square&logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

JARL es un paquete de Python con una arquitectura modular para experimentar con aprendizaje por refuerzo bajo la supervisión de un agente tutor basado en modelos de lenguaje.

El sistema reúne entrenamiento, persistencia y herramientas agénticas. Su gestión de experimentos, inspirada en algunas herramientas de Git, conserva ramas de entrenamiento, configuraciones, métricas, *checkpoints* y artefactos. Cada intervención puede revisarse y relacionarse con la evolución del entrenamiento.

<div align="center">

![Árbol de experimentos de JARL en Runner Lab](docs/assets/images/experiment_tree.png)

</div>

## ✨ Qué permite

- Crear, continuar, bifurcar y comparar entrenamientos sin perder su linaje.
- Entrenar agentes PPO y PPO-GRU sobre Navix mediante JAX.
- Consultar métricas, *checkpoints* y *rollouts* desde Runner Lab.
- Aplicar mediante un agente tutor cambios controlados sobre recompensas, hiperparámetros y tareas.
- Utilizar las mismas operaciones desde Python, la línea de comandos o un cliente MCP.

## 🚀 Inicio rápido

Sigue primero la [guía de instalación](docs/install.es.md). Después, comprueba el dispositivo que utilizará JAX:

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Inicia Runner Lab con:

```bash
make app
```

`make help` muestra los comandos disponibles. `make check` ejecuta las comprobaciones de calidad y `make test` lanza la suite de tests.

## 🤖 Modelos de lenguaje y MCP

Las funciones agénticas requieren un modelo local servido por Ollama o una API compatible con OpenAI. Las APIs remotas necesitan una clave del proveedor. El acceso mediante MCP se realiza desde un cliente compatible, como Cursor, Codex o Claude Code. La [guía de instalación](docs/install.es.md) recoge los requisitos y la configuración básica.

## 📚 Documentación

- [Instalación](docs/install.es.md)
- [Servidor MCP](docs/development/mcp.md)

## 📄 Licencia

JARL se distribuye bajo la [licencia MIT](LICENSE).
