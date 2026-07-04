import { app } from "../../scripts/app.js";

/**
 * MultiLoRALoader stores LoRA rows in node.properties.lora_data and reads
 * ltx_mode from a hidden widget. Coomfy API workflows carry lora_data,
 * ltx_mode, and lora_ui (DOM widget) — inject properties before configure().
 */
app.registerExtension({
    name: "Coomfy.MultiLoRAApiBridge",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MultiLoRALoader") return;

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            const inputs = data?.inputs ?? {};
            const meta = data?._meta ?? {};
            const widgetValues = Array.isArray(data?.widgets_values)
                ? data.widgets_values
                : null;

            const loraPayload =
                inputs.lora_ui ??
                inputs.lora_data ??
                (widgetValues ? widgetValues[0] : undefined);

            if (loraPayload && data && typeof data === "object") {
                data.properties = data.properties || {};
                if (!data.properties.lora_data || data.properties.lora_data === "[]") {
                    data.properties.lora_data =
                        typeof loraPayload === "string"
                            ? loraPayload
                            : JSON.stringify(loraPayload);
                }
            }

            let ltxMode = inputs.ltx_mode ?? meta.ltx_mode;
            if (ltxMode === undefined && widgetValues && widgetValues.length > 1) {
                ltxMode = widgetValues[1];
            }
            if (ltxMode === undefined && (inputs.lora_ui || inputs.lora_data)) {
                ltxMode = true;
            }

            origConfigure?.apply(this, arguments);

            if (loraPayload) {
                this.properties = this.properties || {};
                this.properties.lora_data =
                    typeof loraPayload === "string"
                        ? loraPayload
                        : JSON.stringify(loraPayload);

                const loraWidget = this.widgets?.find((w) => w.name === "lora_data");
                if (loraWidget) {
                    loraWidget.value = this.properties.lora_data;
                }
            }

            if (ltxMode !== undefined) {
                const ltxWidget = this.widgets?.find((w) => w.name === "ltx_mode");
                if (ltxWidget) {
                    ltxWidget.value = !!ltxMode;
                }
            }

            if (this._mllHeader && this._mllRowsContainer && this._mllNodeData) {
                const ltxWidget = this.widgets?.find((w) => w.name === "ltx_mode");
                const checked = !!(ltxWidget?.value ?? ltxMode ?? false);

                const checkbox = this._mllHeader.querySelector(
                    ".mll-ltx-toggle input[type=\"checkbox\"]"
                );
                if (checkbox) {
                    checkbox.checked = checked;
                }
                const toggle = this._mllHeader.querySelector(".mll-ltx-toggle");
                if (toggle) {
                    toggle.classList.toggle("mll-ltx-active", checked);
                }

                if (typeof this.updateLoraData === "function" && this.properties?.lora_data) {
                    this.updateLoraData(this.properties.lora_data);
                }
            }
        };
    },
});
