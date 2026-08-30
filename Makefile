deploy:
	helm upgrade --install label-studio heartex/label-studio -f values.yaml -n private
