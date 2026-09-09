# Navix Maps

Esta página recoge los `env_id` canónicos según la [documentación oficial de Navix](https://epignatelli.com/navix/home/environments.html) y una guía rápida para elegir mapas durante entrenamiento con PPO full-JAX.

La tool `env_navix_maps` y el catálogo del proyecto usan estos nombres canónicos (**69 mapas**). Algunos difieren del registro interno del paquete instalado; JARL resuelve los alias automáticamente al entrenar.

En la instalación local actual del paquete Navix hay **67 mapas** del catálogo disponibles; faltan `Navix-Dynamic-Obstacles-Random-8x8-v0` y `Navix-Dynamic-Obstacles-Random-16x16-v0`. Navix registra además 6 mapas Memory; JARL no los incluye (otro conjunto de acciones). Fetch, GoToObject, PutNear y GoToDoor tampoco entran: la meta va en la misión y PPO no ve ese texto.

## Recomendación rápida

Para PPO con observación `symbolic_first_person` (plana, 147 dims):

```python
"Navix-Empty-5x5-v0"
"Navix-DoorKey-5x5-v0"
"Navix-DoorKey-6x6-v0"
"Navix-DoorKey-8x8-v0"
"Navix-DoorKey-16x16-v0"
```

## Curriculum sugerido

```text
Navix-Empty-5x5-v0
Navix-Empty-8x8-v0
Navix-DoorKey-5x5-v0
Navix-DoorKey-6x6-v0
Navix-DoorKey-8x8-v0
Navix-DoorKey-16x16-v0
```

## Transferencia de checkpoints en JARL

Los 69 `env_id` canónicos comparten el contrato JARL (`symbolic_first_person` plano de 147 dims y 7 acciones: `rotate_ccw`, `rotate_cw`, `forward`, `pickup`, `drop`, `toggle`, `done`). Se pueden intercambiar checkpoints entre ellos, incluidos KeyCorridor, DistShift, Dynamic Obstacles, Crossings, Unlock y ObstructedMaze. `Navix-EmptyVariant-5x5-v0` usa el mismo grupo. Los mapas Memory de Navix quedan fuera de este contrato, igual que Fetch, GoToObject, PutNear y GoToDoor (misión no observable).

Curriculum corto de partida (no es una allowlist):

```python
"Navix-Empty-5x5-v0"
"Navix-Empty-6x6-v0"
"Navix-Empty-8x8-v0"
"Navix-Empty-16x16-v0"
"Navix-DoorKey-5x5-v0"
"Navix-DoorKey-6x6-v0"
"Navix-DoorKey-8x8-v0"
"Navix-DoorKey-16x16-v0"
"Navix-FourRooms-v0"
"Navix-LavaGap-S5-v0"
"Navix-LavaGap-S6-v0"
"Navix-LavaGap-S7-v0"
```

## Distribución (DistShift)

```python
"Navix-DistShift1-v0"
"Navix-DistShift2-v0"
```

## Llave y puerta (DoorKey)

```python
"Navix-DoorKey-5x5-v0"
"Navix-DoorKey-6x6-v0"
"Navix-DoorKey-8x8-v0"
"Navix-DoorKey-16x16-v0"
"Navix-DoorKey-5x5-Random-v0"
"Navix-DoorKey-6x6-Random-v0"
"Navix-DoorKey-8x8-Random-v0"
"Navix-DoorKey-16x16-Random-v0"
```

## Obstáculos dinámicos

```python
"Navix-Dynamic-Obstacles-5x5-v0"
"Navix-Dynamic-Obstacles-6x6-v0"
"Navix-Dynamic-Obstacles-8x8-v0"
"Navix-Dynamic-Obstacles-16x16-v0"
"Navix-Dynamic-Obstacles-Random-5x5-v0"
"Navix-Dynamic-Obstacles-Random-6x6-v0"
"Navix-Dynamic-Obstacles-Random-8x8-v0"
"Navix-Dynamic-Obstacles-Random-16x16-v0"
```

## Navegación simple (Empty)

```python
"Navix-Empty-5x5-v0"
"Navix-Empty-6x6-v0"
"Navix-Empty-8x8-v0"
"Navix-Empty-16x16-v0"
"Navix-Empty-Random-5x5-v0"
"Navix-Empty-Random-6x6-v0"
"Navix-Empty-Random-8x8-v0"
"Navix-Empty-Random-16x16-v0"
```

## Four rooms

```python
"Navix-FourRooms-v0"
"Navix-FourRooms-7x7-v0"
"Navix-FourRooms-9x9-v0"
"Navix-FourRooms-11x11-v0"
"Navix-FourRooms-13x13-v0"
"Navix-FourRooms-15x15-v0"
"Navix-FourRooms-17x17-v0"
```

## Pasillos con llave (KeyCorridor)

```python
"Navix-KeyCorridorS3R1-v0"
"Navix-KeyCorridorS3R2-v0"
"Navix-KeyCorridorS3R3-v0"
"Navix-KeyCorridorS4R3-v0"
"Navix-KeyCorridorS5R3-v0"
"Navix-KeyCorridorS6R3-v0"
```

## Lava gap

```python
"Navix-LavaGap-S5-v0"
"Navix-LavaGap-S6-v0"
"Navix-LavaGap-S7-v0"
```

## Cruces (Crossings)

```python
"Navix-Crossings-S9N1-v0"
"Navix-Crossings-S9N2-v0"
"Navix-Crossings-S9N3-v0"
"Navix-Crossings-S11N5-v0"
```

## Cruces de lava (LavaCrossing)

```python
"Navix-LavaCrossing-S9N1-v0"
"Navix-LavaCrossing-S9N2-v0"
"Navix-LavaCrossing-S9N3-v0"
"Navix-LavaCrossing-S11N5-v0"
```

## Fuera del catálogo (misión no observable)

PPO no recibe el texto de la misión. Estas familias de Navix quedan fuera del catálogo JARL (siguen existiendo en el paquete y en tests de recompensa):

```python
"Navix-GoToDoor-5x5-v0"
"Navix-GoToDoor-6x6-v0"
"Navix-GoToDoor-8x8-v0"
"Navix-GoToObject-6x6-N2-v0"
"Navix-GoToObject-8x8-N2-v0"
"Navix-Fetch-5x5-N2-v0"
"Navix-Fetch-6x6-N2-v0"
"Navix-Fetch-8x8-N3-v0"
"Navix-PutNear-6x6-N2-v0"
"Navix-PutNear-8x8-N3-v0"
```

## Puertas roja y azul (RedBlueDoors)

```python
"Navix-RedBlueDoors-6x6-v0"
"Navix-RedBlueDoors-8x8-v0"
```

## Desbloquear (Unlock)

```python
"Navix-Unlock-v0"
"Navix-UnlockPickup-v0"
"Navix-BlockedUnlockPickup-v0"
```

## Habitación cerrada (LockedRoom)

```python
"Navix-LockedRoom-v0"
```

## Varias habitaciones (MultiRoom)

```python
"Navix-MultiRoom-N2-S4-v0"
"Navix-MultiRoom-N4-S5-v0"
"Navix-MultiRoom-N6-v0"
```

## Laberinto obstruido (ObstructedMaze)

```python
"Navix-ObstructedMaze-1Dl-v0"
"Navix-ObstructedMaze-1Dlh-v0"
"Navix-ObstructedMaze-1Dlhb-v0"
"Navix-ObstructedMaze-2Dl-v0"
"Navix-ObstructedMaze-2Dlh-v0"
"Navix-ObstructedMaze-2Dlhb-v0"
"Navix-ObstructedMaze-1Q-v0"
"Navix-ObstructedMaze-2Q-v0"
"Navix-ObstructedMaze-Full-v0"
```

## Playground

```python
"Navix-Playground-v0"
```

## Memory (fuera del catálogo)

Navix registra `Navix-MemoryS7-v0` … `Navix-MemoryS17Random-v0` con un conjunto de acciones distinto al MiniGrid de 7 acciones. JARL no los incluye en el catálogo canónico ni en el grupo de transferencia.

## Lista completa (69 mapas)

```python
"Navix-DistShift1-v0"
"Navix-DistShift2-v0"
"Navix-DoorKey-5x5-v0"
"Navix-DoorKey-6x6-v0"
"Navix-DoorKey-8x8-v0"
"Navix-DoorKey-16x16-v0"
"Navix-DoorKey-5x5-Random-v0"
"Navix-DoorKey-6x6-Random-v0"
"Navix-DoorKey-8x8-Random-v0"
"Navix-DoorKey-16x16-Random-v0"
"Navix-Dynamic-Obstacles-5x5-v0"
"Navix-Dynamic-Obstacles-6x6-v0"
"Navix-Dynamic-Obstacles-8x8-v0"
"Navix-Dynamic-Obstacles-16x16-v0"
"Navix-Dynamic-Obstacles-Random-5x5-v0"
"Navix-Dynamic-Obstacles-Random-6x6-v0"
"Navix-Dynamic-Obstacles-Random-8x8-v0"
"Navix-Dynamic-Obstacles-Random-16x16-v0"
"Navix-Empty-5x5-v0"
"Navix-Empty-6x6-v0"
"Navix-Empty-8x8-v0"
"Navix-Empty-16x16-v0"
"Navix-Empty-Random-5x5-v0"
"Navix-Empty-Random-6x6-v0"
"Navix-Empty-Random-8x8-v0"
"Navix-Empty-Random-16x16-v0"
"Navix-FourRooms-v0"
"Navix-FourRooms-7x7-v0"
"Navix-FourRooms-9x9-v0"
"Navix-FourRooms-11x11-v0"
"Navix-FourRooms-13x13-v0"
"Navix-FourRooms-15x15-v0"
"Navix-FourRooms-17x17-v0"
"Navix-KeyCorridorS3R1-v0"
"Navix-KeyCorridorS3R2-v0"
"Navix-KeyCorridorS3R3-v0"
"Navix-KeyCorridorS4R3-v0"
"Navix-KeyCorridorS5R3-v0"
"Navix-KeyCorridorS6R3-v0"
"Navix-LavaGap-S5-v0"
"Navix-LavaGap-S6-v0"
"Navix-LavaGap-S7-v0"
"Navix-Crossings-S9N1-v0"
"Navix-Crossings-S9N2-v0"
"Navix-Crossings-S9N3-v0"
"Navix-Crossings-S11N5-v0"
"Navix-LavaCrossing-S9N1-v0"
"Navix-LavaCrossing-S9N2-v0"
"Navix-LavaCrossing-S9N3-v0"
"Navix-LavaCrossing-S11N5-v0"
"Navix-RedBlueDoors-6x6-v0"
"Navix-RedBlueDoors-8x8-v0"
"Navix-Unlock-v0"
"Navix-UnlockPickup-v0"
"Navix-BlockedUnlockPickup-v0"
"Navix-LockedRoom-v0"
"Navix-MultiRoom-N2-S4-v0"
"Navix-MultiRoom-N4-S5-v0"
"Navix-MultiRoom-N6-v0"
"Navix-ObstructedMaze-1Dl-v0"
"Navix-ObstructedMaze-1Dlh-v0"
"Navix-ObstructedMaze-1Dlhb-v0"
"Navix-ObstructedMaze-2Dl-v0"
"Navix-ObstructedMaze-2Dlh-v0"
"Navix-ObstructedMaze-2Dlhb-v0"
"Navix-ObstructedMaze-1Q-v0"
"Navix-ObstructedMaze-2Q-v0"
"Navix-ObstructedMaze-Full-v0"
"Navix-Playground-v0"
```

## Tool agentica

El agente puede **buscar** mapas con `env_navix_maps` (lectura). Casos:

- `env_navix_maps_read`: familia puerta y llave (`categories`).
- `env_navix_maps_search_key`: familias con llave (`door_key` + `key_corridor`; Unlock, LockedRoom y ObstructedMaze también usan llave).
- `env_navix_maps_query_lava`: búsqueda por palabra clave (`query=lava`; LavaGap y LavaCrossing).
- `env_navix_maps_difficulty_easy`: filtro de dificultad (`difficulty_max=20`).
- `env_navix_maps_transfer_ready`: solo transfer JARL (`jarl_transfer_ready_only`).
- `env_navix_maps_filters_combo`: categoría + query + dificultad + transfer a la vez.

El filtro principal es `categories` (lista de slugs exactos, OR). También
acepta `query` (palabras clave sobre id, descripción, label y alias; p. ej.
`llave`, `lava`) y rango `difficulty_min` / `difficulty_max` (heurística 1–100).
Omitir o pasar `[]` en categories lista todo si no hay otros filtros. Si el
agente inventa una key de categoría, la tool responde con las keys reales.

Filtros opcionales:

- `jarl_transfer_ready_only`
- `installed_only`
