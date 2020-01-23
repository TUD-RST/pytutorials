--[[
Reference certain part of the code via a tag and print it using the listings package.

In order to use this function the listings package must be loaded in the LaTeX document.
The advantage of this approach is that you can modify the source code without redefine
the referencing line numbers in the LaTeX document.

Usage: print_code_part{filename}{tagname}

    filename: The name of the file from which parts of the code are going to be listed
    tagname : The tag in the file which marks the part to be listed. See expl. below.

In the source code file mark the part which is going to be shown as follows:

    At begin source code part going to be listed add the following line with a unique tagname
    # LISTING_START tagname

    At the end of the source code part going to be listed add the following line with a unique tagname:
    # LISTING_END tagname

Written by Robert Heedt, Institut für Regelungs- und Steuerungstheorie, TU Dresden, 2019
--]]
function print_code_part(file, listing_tag)
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

    local latex_command = string.format("\\lstinputlisting[numbers=left,firstnumber=%s,firstline=%s,lastline=%s]{%s}", line_start, line_start, line_end, file)
    tex.print(latex_command)
end


--[[
Reference a single line of code via its beginning and print it using the listings package.

In order to use this function the listings package must be loaded in the LaTeX document.
The advantage of this approach is that you can modify the source code without redefine
the referencing line numbers in the LaTeX document.

Usage: print_code_line{filename}{line_starts_with}

    filename: The name of the file from which the code line is going to be listed
    tagname : The string the line begins with

Written by Jan Winkler, Institut für Regelungs- und Steuerungstheorie, TU Dresden, 2020
--]]
function print_code_line(file, line_starts_with)
    local current_line = 0
    local MatchFound = false
    io.input(file)   

    while (MatchFound == false) do
        line_str = io.read("*line")

        if line_str == nil then
            break
        else
            current_line = current_line + 1
        end

        if string.match(line_str, "^" .. line_starts_with) then
            MatchFound = true
        end
    end

    local latex_command = ""
    if MatchFound then
        latex_command = string.format("\\lstinputlisting[numbers=left,firstnumber=%s,firstline=%s,lastline=%s]{%s}", current_line, current_line, current_line, file)
    else
        latex_command = string.format("\\newline \\textbf{No line found starting with \\emph{%s}} \\newline", line_starts_with)
    end
    tex.print(latex_command)
end