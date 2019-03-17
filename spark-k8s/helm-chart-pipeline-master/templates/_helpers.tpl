{{- define "base-fullname" -}}
{{- $name := default .Chart.Name .Values.Base.Name -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

