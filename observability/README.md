## Observability Assets

This directory contains provisioning assets for running a Prometheus + Grafana stack with the application metrics.

### Structure

```
observability/
  grafana/
    dashboards/
      json/
        batch-observability.json   # Main dashboard (import or auto-provision)
    datasources/                   # (optional) provisioning for datasources
  prometheus/
    prometheus.yml                 # (example scrape config)
```

### Prometheus Scrape Config (example)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: fastapi
    metrics_path: /metrics
    static_configs:
      - targets: ['inference-core:8000']
```

### Grafana Provisioning (datasource example)

Place in `observability/grafana/datasources/datasource.yaml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    access: proxy
    isDefault: true
```

### Dashboard Import

1. Open Grafana UI → Dashboards → Import.
2. Upload `batch-observability.json` or copy JSON.
3. Select Prometheus datasource (auto if provisioned).

### Key Panels

- Jobs In Progress (gauge/stat)
- Submission / completion / failure rates
- Provider latency (avg + distribution heatmap)
- Job duration quantiles (p50/p95/p99)
- Poll cycle duration (p95)
- Item success rate (% over last 15m)
- Errors by type & retries by reason
- Recent job status increments (5m window)

### Customization Tips

- Adjust histogram_quantile windows if low traffic (use larger range e.g. `[15m]`).
- Add alerting rules in Prometheus for elevated error or retry rates.
- For long-term retention, extend Prometheus retention or adopt remote storage.

### Notes

- Dashboard UID: `batch-observability` (safe to reference in links).
- Templating variables: `provider`, `operation` (multi-select, All).
- Success rate guards division by zero via `clamp_min(...,1)`.

---

Generated automatically to match current metric names in `inference_core/observability/metrics.py`.
