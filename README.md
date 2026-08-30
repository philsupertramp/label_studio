# Label Studio
Helm chart configuration and helper scripts.

## Installation
```shell
helm upgrade --install label-studio heartex/label-studio -f values.yaml -n private
```

## Configuration
You must update the `app.ingress` configuration inside `values.yaml` to your needs, otherwise
label-studio will not accept any data imports from the same cluster as well as allow you to
authenticate with the service.

### DNS
If you run this service on a single node or with a manual traefik deployment you must set the
```
app:
  hostAliases:
    - ip: "YOUR TRAEFIK HOST"
      hostnames:
        - "YOUR HOST NAMES"
```

## Scripts
We provide two scripts that allow you to 
### Fetch data from a huggingface dataset
`./import_hf.py`
### Annotate images using [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
`./auto_label.py`
