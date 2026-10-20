/** Match the checkbox value type so core's DependNonDefault comparison works. */
postParamBuildSteps.push(() => {
    let param = gen_param_types.find(param => param.id == 'seedvrimagetiling');
    if (param) {
        param.default = `${param.default}` == 'true';
    }
});
