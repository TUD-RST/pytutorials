function print_listing(file, listing_tag)
    local current_line = 1
    local line_start = 1
    local line_end = 1
    io.input(file)

    while true do
        line_str = io.read("*line")

        if line_str == nil then
            line_end = current_line - 1
            break
        -- the thing we're looking for should be at the end of the line or followed by whitespace
        -- this prevents 'tag10' erroneously matching 'tag1'
        elseif string.match(line_str, "LISTING_START " .. listing_tag .. "$") or string.match(line_str, "LISTING_START " .. listing_tag .. "%s") then
            line_start = current_line + 1
        elseif string.match(line_str, "LISTING_END " .. listing_tag .. "$") or string.match(line_str, "LISTING_END " .. listing_tag .. "%s") then
            line_end = current_line - 1
            break
        end

        current_line = current_line + 1
    end

    local latex_command = string.format("\\listcode{%s}{%s}{%s}", line_start, line_end, file)
    tex.print(latex_command)
end