/** Keep declared visibility across Swarm's parameter rebuilds. */
let seedVRTilingVisibility = new WeakMap();

postParamBuildSteps.push(() => {
    let input = getRequiredElementById('input_seedvrimagetiling');
    let children = gen_param_types.filter(param => param.depend_non_default == 'seedvrimagetiling');
    for (let param of children) {
        if (!seedVRTilingVisibility.has(param)) {
            seedVRTilingVisibility.set(param, param.visible);
        }
    }
    let update = () => {
        for (let param of children) {
            param.visible = seedVRTilingVisibility.get(param) && input.checked;
        }
        hideUnsupportableParams();
    };
    input.addEventListener('change', update);
    update();
});
