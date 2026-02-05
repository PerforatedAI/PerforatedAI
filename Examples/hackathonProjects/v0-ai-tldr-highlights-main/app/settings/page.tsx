"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "@/hooks/use-toast"
import { Save, ExternalLink, Check, Copy } from "lucide-react"

export default function SettingsPage() {
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState<string | null>(null)

  const [settings, setSettings] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("forums_settings")
      if (saved) {
        return JSON.parse(saved)
      }
    }
    return {
      apiKey: "",
      instanceId: "",
      instanceHandle: "",
      instanceName: "",
    }
  })

  const handleSave = async () => {
    setSaving(true)
    try {
      // Save to localStorage for client-side persistence
      localStorage.setItem("forums_settings", JSON.stringify(settings))
      toast({
        title: "Settings saved",
        description: "Your Foru.ms configuration has been updated.",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save settings. Please try again.",
        variant: "destructive",
      })
    } finally {
      setSaving(false)
    }
  }

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text)
    setCopied(field)
    setTimeout(() => setCopied(null), 2000)
  }

  return (
    <div className="container mx-auto max-w-4xl p-6 space-y-8">
      <div className="space-y-2">
        <h1 className="font-bold text-3xl">Settings</h1>
        <p className="text-muted-foreground">Configure your Foru.ms instance and API credentials</p>
      </div>

      <Card className="p-6 space-y-6">
        <div className="space-y-2">
          <h2 className="font-semibold text-xl">Foru.ms Configuration</h2>
          <p className="text-muted-foreground text-sm">
            Get your credentials from your{" "}
            <a
              href="https://foru.ms/settings"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline inline-flex items-center gap-1"
            >
              Foru.ms dashboard
              <ExternalLink className="h-3 w-3" />
            </a>
          </p>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="instanceId">Instance ID</Label>
            <div className="flex gap-2">
              <Input
                id="instanceId"
                type="text"
                placeholder="164c1c88-9a97-4335-878c-c0c216280445"
                value={settings.instanceId}
                onChange={(e) => setSettings({ ...settings, instanceId: e.target.value })}
                className="font-mono text-sm"
              />
              <Button variant="outline" size="icon" onClick={() => copyToClipboard(settings.instanceId, "instanceId")}>
                {copied === "instanceId" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <p className="text-muted-foreground text-xs">Unique identifier for your forum instance</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="apiKey">API Key</Label>
            <div className="flex gap-2">
              <Input
                id="apiKey"
                type="password"
                placeholder="06f833a8-b799-4668-8954-e893243c2320"
                value={settings.apiKey}
                onChange={(e) => setSettings({ ...settings, apiKey: e.target.value })}
                className="font-mono text-sm"
              />
              <Button variant="outline" size="icon" onClick={() => copyToClipboard(settings.apiKey, "apiKey")}>
                {copied === "apiKey" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <p className="text-muted-foreground text-xs">Authentication token for API requests</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="instanceHandle">Instance Handle</Label>
            <Input
              id="instanceHandle"
              type="text"
              placeholder="tl-dr"
              value={settings.instanceHandle}
              onChange={(e) => setSettings({ ...settings, instanceHandle: e.target.value })}
              maxLength={16}
            />
            <p className="text-muted-foreground text-xs">
              URL-friendly identifier (max 16 characters, alphanumeric + hyphens)
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="instanceName">Instance Name</Label>
            <Input
              id="instanceName"
              type="text"
              placeholder="tl dr"
              value={settings.instanceName}
              onChange={(e) => setSettings({ ...settings, instanceName: e.target.value })}
              maxLength={16}
            />
            <p className="text-muted-foreground text-xs">Display name for your forum (max 16 characters)</p>
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t">
          <Button variant="outline" onClick={() => window.location.reload()}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            <Save className="mr-2 h-4 w-4" />
            {saving ? "Saving..." : "Save Settings"}
          </Button>
        </div>
      </Card>

      <Card className="p-6 space-y-4 bg-muted/50">
        <h3 className="font-semibold">Quick Setup Guide</h3>
        <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
          <li>Visit your Foru.ms dashboard and navigate to Settings</li>
          <li>Copy your Instance ID and API Key from the General Settings section</li>
          <li>Paste the credentials into the fields above</li>
          <li>Set your instance handle and name</li>
          <li>Click "Save Settings" to apply the configuration</li>
        </ol>
      </Card>
    </div>
  )
}
