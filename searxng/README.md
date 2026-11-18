```bash exec="1" source="block"
cat << 'EOF' > /tmp/searxng-config/settings.yml
use_default_settings: true

general:
  contact_url: "mailto:ai@df-trading.ru"

search:
  formats:
    - html
    - json

server:
  secret_key: "super-random-secret-key-change-me"
  limiter: false
EOF

docker run -d --name searxng-test \
  -p 8888:8080 \
  -v /tmp/searxng-config:/etc/searxng \
  -e SEARXNG_BASE_URL="http://192.168.27.80:8888/" \
  searxng/searxng:latest
```
