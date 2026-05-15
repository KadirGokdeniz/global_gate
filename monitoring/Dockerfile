FROM grafana/alloy:latest

RUN mkdir -p /etc/alloy && cat > /etc/alloy/config.alloy << 'ALLOY_EOF'
prometheus.scrape "fastapi_rag" {
  targets = [{
    __address__ = "globalgate-production.up.railway.app",
  }]

  metrics_path    = "/metrics"
  scheme          = "https"
  scrape_interval = "15s"
  scrape_timeout  = "10s"

  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = "https://prometheus-prod-65-prod-eu-west-2.grafana.net/api/prom/push"

    basic_auth {
      username = "3211484"
      password = env("GRAFANA_API_KEY")
    }
  }
}
ALLOY_EOF

EXPOSE 12345
CMD ["/bin/alloy", "run", "--server.http.listen-addr=0.0.0.0:12345", "/etc/alloy/config.alloy"]