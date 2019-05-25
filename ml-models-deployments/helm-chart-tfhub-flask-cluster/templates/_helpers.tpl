{{- define "tfhub-flask-fullname" -}}
{{- $name := default .Chart.Name .Values.TFHubFlask.Name -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

